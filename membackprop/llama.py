"""Llama 3.x written from scratch in PyTorch, loading Hugging Face weights."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class RopeScaling:
    """Llama 3.1+ frequency scaling parameters."""

    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_context: int


@dataclass(frozen=True)
class LlamaConfig:
    """Architecture hyperparameters."""

    vocab_size: int
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    norm_eps: float
    rope_theta: float
    tie_embeddings: bool
    rope_scaling: RopeScaling | None = None

    @classmethod
    def from_hf(cls, raw: dict) -> "LlamaConfig":
        """Build a config from a Hugging Face ``config.json`` dictionary."""
        # Newer transformers nests theta and scaling under rope_parameters.
        rope = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
        scaling = None
        if rope.get("rope_type") == "llama3":
            scaling = RopeScaling(
                rope["factor"],
                rope["low_freq_factor"],
                rope["high_freq_factor"],
                rope["original_max_position_embeddings"],
            )
        return cls(
            vocab_size=raw["vocab_size"],
            dim=raw["hidden_size"],
            hidden_dim=raw["intermediate_size"],
            n_layers=raw["num_hidden_layers"],
            n_heads=raw["num_attention_heads"],
            n_kv_heads=raw["num_key_value_heads"],
            head_dim=raw.get("head_dim")
            or raw["hidden_size"] // raw["num_attention_heads"],
            norm_eps=raw["rms_norm_eps"],
            rope_theta=raw.get("rope_theta") or rope["rope_theta"],
            tie_embeddings=raw["tie_word_embeddings"],
            rope_scaling=scaling,
        )


class RMSNorm(nn.Module):
    """Root-mean-square normalization."""

    def __init__(self, dim: int, eps: float):
        """Initialize the gain to one."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalize over the last dimension in float32."""
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.type_as(x)


def rope_frequencies(cfg: LlamaConfig) -> Tensor:
    """Compute inverse RoPE frequencies, applying Llama 3.1 scaling if set.

    Args:
        cfg: Model configuration.

    Returns:
        Tensor of shape ``(head_dim / 2,)``.
    """
    exponents = torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim
    freqs = cfg.rope_theta**-exponents
    if cfg.rope_scaling is None:
        return freqs
    s = cfg.rope_scaling
    # Rotations per original context: few means low frequency, so scale it fully.
    cycles = s.original_context * freqs / (2 * math.pi)
    blend = (cycles - s.low_freq_factor) / (s.high_freq_factor - s.low_freq_factor)
    blend = blend.clamp(0, 1)
    return blend * freqs + (1 - blend) * freqs / s.factor


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate ``(batch, heads, seq, head_dim)`` features by position."""
    a, b = x.chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


class Attention(nn.Module):
    """Grouped-query causal self-attention."""

    def __init__(self, cfg: LlamaConfig):
        """Create the four projections."""
        super().__init__()
        self.head_dim = cfg.head_dim
        self.groups = cfg.n_heads // cfg.n_kv_heads
        kv_dim = cfg.n_kv_heads * cfg.head_dim
        q_dim = cfg.n_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.dim, q_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, cfg.dim, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Attend causally over the sequence."""
        b, t, _ = x.shape
        q, k, v = (
            proj(x).view(b, t, -1, self.head_dim).transpose(1, 2)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class MLP(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, cfg: LlamaConfig):
        """Create the gate, up and down projections."""
        super().__init__()
        self.gate_proj = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.down_proj = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the gated transformation."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """Pre-norm transformer layer."""

    def __init__(self, cfg: LlamaConfig):
        """Create attention, MLP and their norms."""
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply attention then MLP, each with a residual connection."""
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Llama(nn.Module):
    """Llama decoder; ``forward`` stops at hidden states so losses can chunk."""

    def __init__(self, cfg: LlamaConfig):
        """Create the decoder stack and output head."""
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.register_buffer("inv_freq", rope_frequencies(cfg), persistent=False)

    def forward(self, tokens: Tensor) -> Tensor:
        """Return final hidden states of shape ``(batch, seq, dim)``."""
        x = self.embed_tokens(tokens)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        angles = torch.outer(positions.float(), self.inv_freq)
        cos, sin = angles.cos().type_as(x), angles.sin().type_as(x)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)

    @classmethod
    def from_pretrained(cls, path: str | Path, dtype=torch.float32) -> "Llama":
        """Load a local Hugging Face Llama checkpoint directory.

        Args:
            path: Directory with ``config.json`` and safetensors shards.
            dtype: Parameter dtype.

        Returns:
            The loaded model.
        """
        from safetensors.torch import load_file

        path = Path(path)
        cfg = LlamaConfig.from_hf(json.loads((path / "config.json").read_text()))
        state = {}
        for shard in sorted(path.glob("*.safetensors")):
            state.update(load_file(shard))
        model = cls(cfg)
        model.load_hf_state_dict(state)
        return model.to(dtype)

    def load_hf_state_dict(self, state: dict[str, Tensor]) -> None:
        """Copy weights from a Hugging Face Llama state dict, strictly."""
        renamed = {k.removeprefix("model."): v for k, v in state.items()}
        if self.lm_head.weight is self.embed_tokens.weight:
            renamed["lm_head.weight"] = renamed["embed_tokens.weight"]
        self.load_state_dict(renamed, strict=True)
