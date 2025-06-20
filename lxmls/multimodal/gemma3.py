import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

# TODO: Bring into file
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, make_nested_list_of_images
from transformers.processing_utils import (
    ImagesKwargs,
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import to_py_obj


# Config
@dataclass
class Gemma3TextConfig:
    vocab_size: int = 32000
    hidden_size: int = 2560
    intermediate_size: int = 15360
    num_hidden_layers: int = 28
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_activation: str = "silu"
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    rope_local_base_freq: float = 500000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    query_pre_attn_scalar: float = 28.0
    attn_logit_softcapping: Optional[float] = 50.0
    final_logit_softcapping: Optional[float] = 30.0
    sliding_window: int = 4096
    layer_types: List[str] = field(default_factory=lambda: ["full_attention"] * 28)
    pad_token_id: int = 0

    def __post_init__(self):
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"`layer_types` must have a length of {self.num_hidden_layers}."
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "`hidden_size` must be divisible by `num_attention_heads`."
            )


@dataclass
class Gemma3VisionConfig:
    model_name: str = "google/siglip-base-patch16-224"
    layer_norm_eps: float = 1e-6


@dataclass
class Gemma3Config:
    text_config: Gemma3TextConfig = field(default_factory=Gemma3TextConfig)
    vision_config: Gemma3VisionConfig = field(default_factory=Gemma3VisionConfig)

    image_token_id: int = 32001
    mm_tokens_per_image: int = 256

    def __post_init__(self):
        if self.text_config.vocab_size <= self.image_token_id:
            self.text_config.vocab_size = self.image_token_id + 1


# Helper fns
def _get_activation_fn(name: str):
    if name == "silu":
        return F.silu
    if name == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported activation function: {name}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    q_embed, k_embed = (
        (q * cos) + (rotate_half(q) * sin),
        (k * cos) + (rotate_half(k) * sin),
    )
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def create_causal_mask(
    seq_len: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    mask = torch.full(
        (seq_len, seq_len),
        dtype=dtype,
        device=device,
        fill_value=torch.finfo(dtype).min,
    )
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :]


def create_sliding_window_causal_mask(
    seq_len: int, window_size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    mask = torch.full(
        (seq_len, seq_len),
        dtype=dtype,
        device=device,
        fill_value=torch.finfo(dtype).min,
    )
    mask_cond = (
        torch.arange(seq_len, device=device)[None, :]
        > torch.arange(seq_len, device=device)[:, None]
    )
    mask.masked_fill_(mask_cond, 0)
    if window_size > 0:
        window_mask = torch.arange(seq_len, device=device)[None, :] < (
            torch.arange(seq_len, device=device)[:, None] + window_size
        )
        mask.masked_fill_(~window_mask, torch.finfo(dtype).min)
    return mask[None, None, :, :]


# Layers
class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()) * (1.0 + self.weight.float())
        return output.type_as(x)


class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        vision_hf_config = AutoConfig.from_pretrained(config.vision_config.model_name)

        vision_hidden_size = vision_hf_config.hidden_size
        text_hidden_size = config.text_config.hidden_size

        self.mm_input_projection_weight = nn.Parameter(
            torch.randn(vision_hidden_size, text_hidden_size)
        )
        self.mm_soft_emb_norm = Gemma3RMSNorm(
            vision_hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = (
            vision_hf_config.image_size // vision_hf_config.patch_size
        )
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.kernel_size, stride=self.kernel_size
        )

    def forward(self, vision_outputs: torch.Tensor):
        # vision_outputs: [batch_size, num_patches + 1, vision_hidden_size]
        # Discard the CLS token for pooling, as is common practice
        vision_outputs = vision_outputs[:, 1:, :]
        batch_size, _, hidden_size = vision_outputs.shape

        vision_outputs_reshaped = vision_outputs.transpose(1, 2).reshape(
            batch_size, hidden_size, self.patches_per_image, self.patches_per_image
        )

        pooled_outputs = self.avg_pool(vision_outputs_reshaped)
        pooled_outputs = pooled_outputs.flatten(2).transpose(1, 2)
        normed_outputs = self.mm_soft_emb_norm(pooled_outputs)
        projected_outputs = torch.matmul(
            normed_outputs, self.mm_input_projection_weight
        )

        return projected_outputs.type_as(vision_outputs)


class Gemma3TextScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = math.sqrt(embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale


class Gemma3RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        self.dim, self.max_position_embeddings, self.base = (
            dim,
            max_position_embeddings,
            base,
        )
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(
            max_position_embeddings, self.inv_freq.device, torch.get_default_dtype()
        )

    def _set_cos_sin_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = position_ids.max() + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        cos = self.cos_cached.index_select(0, position_ids.reshape(-1)).view(
            *position_ids.shape, -1
        )
        sin = self.sin_cached.index_select(0, position_ids.reshape(-1)).view(
            *position_ids.shape, -1
        )
        return cos, sin


class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = _get_activation_fn(config.hidden_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma3Attention(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.num_heads, self.num_kv_heads = (
            config.num_attention_heads,
            config.num_key_value_heads,
        )
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_value=None,
    ):
        bsz, q_len, _ = hidden_states.size()
        query_states, key_states, value_states = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        query_states, key_states = self.q_norm(query_states), self.k_norm(key_states)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            key_states, value_states = (
                torch.cat([past_key_value[0], key_states], dim=2),
                torch.cat([past_key_value[1], value_states], dim=2),
            )
        present_key_value = (key_states, value_states)

        key_states, value_states = (
            repeat_kv(key_states, self.num_kv_groups),
            repeat_kv(value_states, self.num_kv_groups),
        )
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        )

        if self.attn_logit_softcapping is not None:
            attn_weights = (
                torch.tanh(attn_weights / self.attn_logit_softcapping)
                * self.attn_logit_softcapping
            )
        if attention_mask is not None:
            kv_seq_len = key_states.shape[-2]
            attn_weights = attn_weights + attention_mask[:, :, -q_len:, -kv_seq_len:]

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = (
            torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        )
        attn_output = self.o_proj(attn_output.reshape(bsz, q_len, -1))
        return attn_output, attn_weights, present_key_value


class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Gemma3Attention(config, layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states,
        position_embeddings_global,
        position_embeddings_local,
        attention_mask=None,
        past_key_value=None,
    ):
        residual = hidden_states
        hidden_states_norm = self.input_layernorm(hidden_states)
        position_embeddings = (
            position_embeddings_local
            if self.self_attn.is_sliding
            else position_embeddings_global
        )

        attn_output, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states_norm,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
        )
        hidden_states = residual + self.post_attention_layernorm(attn_output)

        residual = hidden_states
        hidden_states_norm = self.pre_feedforward_layernorm(hidden_states)
        hidden_states_mlp = self.mlp(hidden_states_norm)
        hidden_states = residual + self.post_feedforward_layernorm(hidden_states_mlp)

        return hidden_states, self_attn_weights, present_key_value


# Wrappers
class Gemma3TextModel(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.rotary_emb_local = Gemma3RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_local_base_freq
        )
        self.causal_mask = create_causal_mask(
            config.max_position_embeddings, None, torch.get_default_dtype()
        )
        self.sliding_window_mask = create_sliding_window_causal_mask(
            config.max_position_embeddings,
            config.sliding_window,
            None,
            torch.get_default_dtype(),
        )

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify one of `input_ids` or `inputs_embeds`.")

        hidden_states = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        bsz, seq_len, _ = hidden_states.shape

        if position_ids is None:
            past_kv_len = (
                past_key_values[0][0].shape[2] if past_key_values is not None else 0
            )
            position_ids = torch.arange(
                past_kv_len,
                past_kv_len + seq_len,
                dtype=torch.long,
                device=hidden_states.device,
            ).unsqueeze(0)

        pos_emb_global, pos_emb_local = (
            self.rotary_emb(hidden_states, position_ids),
            self.rotary_emb_local(hidden_states, position_ids),
        )
        next_kv_cache, all_attentions = [], []

        for i, decoder_layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            attn_mask = (
                self.sliding_window_mask
                if self.config.layer_types[i] == "sliding_attention"
                else self.causal_mask
            )

            hidden_states, attn_weights, present_key_value = decoder_layer(
                hidden_states,
                pos_emb_global,
                pos_emb_local,
                attention_mask=attn_mask.to(hidden_states.device, hidden_states.dtype),
                past_key_value=past_kv,
            )
            if past_key_values is not None:
                next_kv_cache.append(present_key_value)
            if output_attentions:
                all_attentions.append(attn_weights)

        hidden_states = self.norm(hidden_states)
        return hidden_states, next_kv_cache or None, all_attentions or None


class Gemma3ForConditionalGeneration(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.vision_tower = AutoModel.from_pretrained(config.vision_config.model_name)
        self.multi_modal_projector = Gemma3MultiModalProjector(config)
        self.language_model = Gemma3TextModel(config.text_config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def forward(
        self,
        input_ids,
        pixel_values=None,
        labels=None,
        past_key_values=None,
        output_attentions=False,
    ):
        text_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            # Use the standard output format from Hugging Face models
            image_features = self.vision_tower(pixel_values).last_hidden_state
            projected_image_features = self.multi_modal_projector(image_features)

            image_token_mask = input_ids == self.config.image_token_id
            num_images = image_token_mask.sum()

            if num_images > 0:
                if num_images != projected_image_features.view(-1, 1).shape[0]:
                    raise ValueError(
                        "Number of image tokens in text does not match the number of projected image features."
                    )

                text_embeds[image_token_mask] = projected_image_features.view(
                    -1, text_embeds.shape[-1]
                )

        hidden_states, next_kv_cache, all_attentions = self.language_model(
            inputs_embeds=text_embeds,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
        )

        logits = self.lm_head(hidden_states)
        if self.config.text_config.final_logit_softcapping is not None:
            logits = (
                torch.tanh(logits / self.config.text_config.final_logit_softcapping)
                * self.config.text_config.final_logit_softcapping
            )

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = (
                logits[..., :-1, :]
                .contiguous()
                .view(-1, self.config.text_config.vocab_size)
            )
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = loss_fct(shift_logits, shift_labels.to(shift_logits.device))

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": next_kv_cache,
            "attentions": all_attentions,
        }


# Processor
class Gemma3ImagesKwargs(ImagesKwargs):
    do_pan_and_scan: Optional[bool]
    pan_and_scan_min_crop_size: Optional[int]
    pan_and_scan_max_num_crops: Optional[int]
    pan_and_scan_min_ratio_to_activate: Optional[float]
    do_convert_rgb: Optional[bool]


class Gemma3ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Gemma3ImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": True,
        },
        "images_kwargs": {
            "do_pan_and_scan": False,
            "pan_and_scan_min_crop_size": 256,
            "pan_and_scan_max_num_crops": 4,
            "pan_and_scan_min_ratio_to_activate": 1.2,
        },
    }


class Gemma3Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor,
        tokenizer,
        chat_template=None,
        image_seq_length: int = 256,
        **kwargs,
    ):
        self.image_seq_length = image_seq_length
        self.image_token_id = tokenizer.image_token_id
        self.boi_token = tokenizer.boi_token
        self.image_token = tokenizer.image_token
        image_tokens_expanded = "".join([tokenizer.image_token] * image_seq_length)
        self.full_image_sequence = (
            f"\n\n{tokenizer.boi_token}{image_tokens_expanded}{tokenizer.eoi_token}\n\n"
        )

        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[
            TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]
        ] = None,
        videos=None,
        audio=None,
        **kwargs: Unpack[Gemma3ProcessorKwargs],
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("Provide at least one of `text` or `images`.")

        output_kwargs = self._merge_kwargs(
            Gemma3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError(
                "Invalid input text. Please provide a string, or a list of strings"
            )

        image_inputs = {}
        if images is not None:
            batched_images = make_nested_list_of_images(images)
            image_inputs = self.image_processor(
                batched_images, **output_kwargs["images_kwargs"]
            )

            # Create empty text to be replaced with placeholders
            if not text:
                text = [
                    " ".join([self.boi_token] * len(images))
                    for images in batched_images
                ]

            if len(batched_images) != len(text):
                raise ValueError(
                    f"Received inconsistently sized batches of images ({len(batched_images)}) and text ({len(text)})."
                )

            # Replace image tokens by the full expanded sequence
            num_crops = to_py_obj(image_inputs.pop("num_crops"))
            batch_num_crops = [
                [num_crops.pop(0) for _ in range(len(images))]
                for images in batched_images
            ]
            for batch_idx, (prompt, images, num_crops) in enumerate(
                zip(text, batched_images, batch_num_crops)
            ):
                image_indexes = [m.start() for m in re.finditer(self.boi_token, prompt)]

                if len(images) != len(image_indexes):
                    raise ValueError(
                        f"Prompt contained {len(image_indexes)} image tokens but received {len(images)} images."
                    )

                # Insert additional image tokens for Pan-and-Scan crops
                for num, idx in reversed(list(zip(num_crops, image_indexes))):
                    if num:
                        formatted_image_text = (
                            f"Here is the original image {self.boi_token} and here are some crops to help you see better "
                            + " ".join([self.boi_token] * num)
                        )
                        prompt = (
                            prompt[:idx]
                            + formatted_image_text
                            + prompt[idx + len(self.boi_token) :]
                        )
                        text[batch_idx] = prompt

            # Expand placeholder image tokens to the full image token sequence
            text = [
                prompt.replace(self.boi_token, self.full_image_sequence)
                for prompt in text
            ]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop(
            "return_mm_token_type_ids", False
        )
        text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        # Add token type ids manually, as tokenizer can't do arbitrary position token types
        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(array_ids)
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(
            data={**text_inputs, **image_inputs}, tensor_type=return_tensors
        )

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.

        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """

        vision_data = {}
        if image_sizes is not None:
            # NOTE: no image cropping supported yet
            num_image_tokens = [self.image_seq_length] * len(image_sizes)
            num_image_patches = [1] * len(image_sizes)

            vision_data.update(
                {
                    "num_image_tokens": num_image_tokens,
                    "num_image_patches": num_image_patches,
                }
            )

        return MultiModalData(**vision_data)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Gemma
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Gemma
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names + ["token_type_ids"]
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


def inference():
    model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-it")
    processor = Gemma3Processor.from_pretrained("google/gemma-3-4b-it")

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                },
                {"type": "text", "text": "Where is the cat standing?"},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    generate_ids = model.generate(**inputs)
    return processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
