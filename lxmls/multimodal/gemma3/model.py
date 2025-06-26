import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import (
    AutoTokenizer,
    DynamicCache,  # TODO Free from HF
)
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask  # TODO Free from HF

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)7s - %(message)s")
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.setLevel(logging.INFO)


# https://huggingface.co/google/gemma-3-4b-it/blob/main/config.json
class Gemma3TextConfig:
    def __init__(self):
        self._attn_implementation: str = "eager"  # TODO free from hf

        self.vocab_size: int = 262_208

        self.num_hidden_layers: int = 34
        self.hidden_size: int = 2560
        self.intermediate_size: int = 10240
        self.max_position_embeddings: int = 131_072

        self.pad_token_id: int = 0
        self.eos_token_id: int = 1
        self.bos_token_id: int = 2

        self.rms_norm_eps: float = 1e-6

        self.h = 8
        self.h_kv: int = 4
        self.head_dim: int = 256
        self.sliding_window: int = 4096

        self.layer_types: list[str] = [
            "sliding_attention" if bool((i + 1) % 6) else "full_attention" for i in range(self.num_hidden_layers)
        ]
        self.query_pre_attn_scalar: int = 256
        self.attention_dropout: float = 0.0
        self.attention_bias: bool = False
        self.attn_logit_softcapping: Optional[float] = None

        self.rope_theta = 1_000_000.0
        self.rope_local_base_freq = 10_000.0
        self.rope_scaling = {
            "factor": 8.0,
            "rope_type": "linear",
        }


class ScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        # https://github.com/pytorch/pytorch/issues/18056#issue-421475751
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input: torch.Tensor):
        assert isinstance(self.embed_scale, torch.Tensor)
        return super().forward(input) * self.embed_scale.to(self.weight.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # https://github.com/huggingface/transformers/pull/29402
        x = x.float()
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        out = out * (1.0 + self.weight.float())  # Note: out * (out + gamma)
        return out.type_as(x)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # https://x.com/giffmana/status/1763655413088530697
        self.act_fn = partial(F.gelu, approximate="tanh")

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, rope_theta, rope_scaling, head_dim, device=None):
        super().__init__()
        rope_type = rope_scaling.get("rope_type", "default")

        if rope_type == "default":
            inv_freq = self._init_default(rope_theta=rope_theta, head_dim=head_dim, device=device)
        elif rope_type == "linear":
            inv_freq = self._init_linear(
                rope_theta=rope_theta, head_dim=head_dim, factor=rope_scaling.get("factor"), device=device
            )
        else:
            raise NotImplementedError()

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    def _init_default(self, rope_theta, head_dim: int, device=None):
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / head_dim)
        )
        return inv_freq

    def _init_linear(self, rope_theta, head_dim: int, factor: float, device=None):
        inv_freq = self._init_default(rope_theta=rope_theta, head_dim=head_dim, device=device)
        inv_freq /= factor
        return inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        assert isinstance(self.inv_freq, torch.Tensor)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


class Attention(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int, training: bool = False):
        super().__init__()
        self.training = training

        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.dropout = config.attention_dropout
        self.softcap = config.attn_logit_softcapping
        self.layer_idx = layer_idx

        self.scaling = config.query_pre_attn_scalar**-0.5

        self.head_dim = config.head_dim
        self.num_kv_groups = config.h // config.h_kv

        self.q_proj = nn.Linear(config.hidden_size, config.h * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.h_kv * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.h_kv * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.h * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)

    # FIXME Maybe use F.sdpa
    def _attn(self, q, k, v, mask, dropout, scaling=None, softcap=None, sliding_window=None):
        if scaling is None:
            scaling = self.head_dim**-0.5

        repeat_kv = partial(torch.repeat_interleave, dim=1)
        k = repeat_kv(k, repeats=self.num_kv_groups)
        v = repeat_kv(v, repeats=self.num_kv_groups)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling

        if softcap is not None:
            attn_weights = torch.tanh(attn_weights / softcap) * softcap
        if mask is not None:
            attn_weights = attn_weights + mask[:, :, :, : k.shape[-2]]

        # Upcast to fp32
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=self.training)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous()

        return attn_out

    def forward(self, x, pe, mask, past_kv, cache_pos):
        logger.debug(f"Attention.forward START - {x.shape}")
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project and divide across heads
        q = self.q_proj(x).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(x).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        # Pre-attention layernorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        cos, sin = pe
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Update cache
        if past_kv is not None:
            # TODO Implement update
            k, v = past_kv.update(k, v, self.layer_idx, cache_pos)

        # Attention
        attn_out = self._attn(q, k, v, mask, self.dropout, self.scaling, self.sliding_window)
        attn_out = attn_out.reshape(*input_shape, -1).contiguous()
        attn_out = self.o_proj(attn_out)

        logger.debug(f"Attention.forward END - {x.shape}")
        return attn_out


class DecoderLayer(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Attention(config=config, layer_idx=layer_idx)

        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, pe_global, pe_local, attention_mask, position_ids=None, past_kv=None, cache_pos=None):
        logger.debug(f"DecoderLayer[{self.layer_idx}].forward START - {x.shape}")
        residual = x

        x = self.input_layernorm(x)

        if self.self_attn.is_sliding:
            pe = pe_local
        else:
            pe = pe_global

        x = self.self_attn(x=x, pe=pe, mask=attention_mask, past_kv=past_kv, cache_pos=cache_pos)
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        logger.debug(f"DecoderLayer[{self.layer_idx}].forward END - {x.shape}")
        return x


class Gemma3TextModel(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )

        self.layers = nn.ModuleList([DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rope_global = RotaryEmbedding(
            rope_theta=config.rope_theta, rope_scaling=config.rope_scaling, head_dim=config.head_dim
        )
        self.rope_local = RotaryEmbedding(
            rope_theta=config.rope_local_base_freq, rope_scaling={"rope_type": "default"}, head_dim=config.head_dim
        )

    def forward(self, input_ids, attention_mask, position_ids=None, past_kv=None, cache_pos=None):
        logger.debug("Gemma3TextModel.forward START")
        input_embeds = self.embed_tokens(input_ids)

        if past_kv is None:
            past_kv = DynamicCache()  # TODO Free from HF

        if cache_pos is None:
            past_seen_tokens = past_kv.get_seq_length() if past_kv is not None else 0
            cache_pos = torch.arange(
                past_seen_tokens, past_seen_tokens + input_embeds.shape[1], device=input_embeds.device
            )

        if position_ids is None:
            position_ids = cache_pos.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": input_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_kv,
                "cache_position": cache_pos,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        x = input_embeds
        pe_global = self.rope_global(x, position_ids)
        pe_local = self.rope_local(x, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            assert isinstance(decoder_layer.attention_type, str)
            logger.debug(f"SHAPE: {x.shape}")
            x = decoder_layer(
                x,
                pe_global=pe_global,
                pe_local=pe_local,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_kv=past_kv,
                cache_pos=cache_pos,
            )

        x = self.norm(x)

        logger.debug("Gemma3TextModel.forward END")
        return x, past_kv


class Gemma3ForCausalLM(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.model = Gemma3TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(self.model.embed_tokens.weight)  # FIXME is transpose needed?

    def forward(self, input_ids, attention_mask, position_ids=None, past_kv=None, cache_pos=None):
        logger.debug("Gemma3ForCausalLM.forward START")
        out, _ = self.model(input_ids, attention_mask, position_ids, past_kv, cache_pos)
        logits = self.lm_head(out[:, slice(0, None), :])
        logger.debug("Gemma3ForCausalLM.forward END")
        return logits

    def load_weights(self, shards_dir: Path | str):
        logger.debug(f"Loading weights from {shards_dir}")
        shards_dir = Path(shards_dir)
        state_dict = {}
        for shard_file in sorted(list(shards_dir.glob("*.safetensors"))):
            shard = load_file(shard_file, device="cpu")
            state_dict.update(shard)
        lm_state_dict = {
            k[len("language_model.") :]: v for k, v in state_dict.items() if k.startswith("language_model.")
        }
        self.load_state_dict(lm_state_dict, strict=False)


def decode(logits: torch.Tensor, tokenizer, temperature: float = 0.8):
    logger.debug("Decoding logits")
    # Get logits of last token
    # NOTE Hardcoded batch=0
    x = logits[0, -1, :]
    x /= temperature
    probs = F.softmax(x, dim=-1)
    idx = torch.argmax(probs, dim=-1).detach().cpu()
    return tokenizer.decode(idx, skip_special_tokens=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--shards", type=str)
    parser.add_argument("--len", type=int, default=10)
    parser.add_argument("--prompt", type=str, default="Write a poem about a chonky cat.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debugging")

    logger.debug("Initialising Gemma3TextConfig")
    config = Gemma3TextConfig()

    logger.debug("Initialising Gemma3ForCausalLM")
    model = Gemma3ForCausalLM(config)
    model.load_weights(args.shards)

    logger.debug("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

    prompt = args.prompt
    for _ in range(args.len):
        inputs = tokenizer(prompt, return_tensors="pt")
        logits = model(**inputs)
        gen = decode(logits, tokenizer)
        print(gen, end="", flush=True)
        prompt += gen
