"""nanoGPT-style transformer: RoPE, RMSNorm, no biases, tied embeddings (SPEC §1.2).

This model conforms to `accounting.py`, not the other way round. `count_weight_bits` is the
experiment's x-axis, so a discrepancy between the formula and what actually gets
instantiated would make every number wrong in a way nothing else would catch.
A test (not part of this release) asserts the match for every
configuration in the grid.

Structural choices, all forced by SPEC §1.2 rather than chosen:

* **RoPE**, not learned positions — learned positions would consume budget and vary with
  width, adding a confound to the very axis under study. RoPE is free.
* **No biases anywhere.** Same reason, plus they are unquantized parameters that would
  need their own accounting.
* **Tied embedding/head**, counted once.
* **Two-matrix MLP** (up, GELU, down), not SwiGLU: SPEC §1.3's parameter formula is
  ``2 * mlp_ratio * n_embd**2``, which is exactly two matrices. SwiGLU's three would break
  the accounting.

Dtype policy (SPEC §2.4, weights-only quantization):

* Latent weights are fp32 everywhere, including RMSNorm gains. Quantization happens in the
  forward pass each step.
* Activations are bf16. The RMSNorm reduction and the final cross-entropy run in fp32 for
  numerical stability — standard practice, applied identically to every run.
* SPEC §1.2's "gains kept in bf16 and counted" is a statement about *deployed storage*, and
  that is how `accounting.py` counts them: 16 bits each. Keeping the latent gain in fp32
  matches the treatment of every other weight, whose latent is also fp32 while being
  counted at ``b`` bits.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from isomem.accounting import ModelConfig, count_params, norm_params
from isomem.quant import BF16_CONTROL_BITS, QuantLinear

RMSNORM_EPS = 1e-5


class RMSNorm(nn.Module):
    """RMSNorm with a learned gain and no bias. Reduction in fp32 for stability."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + RMSNORM_EPS)
        return (normed * self.weight).to(x.dtype)


def rope_tables(
    head_dim: int, block_size: int, theta: float, device, dtype=torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precomputed cos/sin for rotary embeddings. Shape (block_size, head_dim)."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(block_size, device=device).float()
    freqs = torch.outer(pos, inv_freq)              # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)         # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, n_head, T, head_dim); cos/sin: (T, head_dim)."""
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return x * cos + _rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    """Fused QKV projection and output projection. 4 * n_embd**2 parameters."""

    def __init__(self, cfg: ModelConfig, bits: float):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.qkv = QuantLinear(cfg.n_embd, 3 * cfg.n_embd, bits)
        self.proj = QuantLinear(cfg.n_embd, cfg.n_embd, bits)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    """Two matrices and a GELU. 2 * mlp_ratio * n_embd**2 parameters (SPEC §1.3)."""

    def __init__(self, cfg: ModelConfig, bits: float):
        super().__init__()
        self.up = QuantLinear(cfg.n_embd, cfg.mlp_hidden, bits)
        self.down = QuantLinear(cfg.mlp_hidden, cfg.n_embd, bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    """Pre-norm transformer block. Two RMSNorm gains, per the accounting."""

    def __init__(self, cfg: ModelConfig, bits: float):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg, bits)
        self.norm_mlp = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg, bits)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), cos, sin)
        return x + self.mlp(self.norm_mlp(x))


class IsoMemGPT(nn.Module):
    """The model under test. One bit-width for every quantized weight."""

    def __init__(self, cfg: ModelConfig, bits: float, rope_theta: float = 10000.0):
        super().__init__()
        self.cfg = cfg
        self.bits = bits
        self.rope_theta = rope_theta

        # One tied matrix serving as both the token embedding and the output head,
        # counted once (SPEC §1.2). Held as a QuantLinear so it is quantized like any
        # other weight -- at small budgets it IS most of the budget (SPEC §2.10).
        self.embed = QuantLinear(cfg.n_embd, cfg.vocab_size, bits)
        self.blocks = nn.ModuleList([Block(cfg, bits) for _ in range(cfg.n_layer)])
        self.norm_final = RMSNorm(cfg.n_embd)

        cos, sin = rope_tables(cfg.head_dim, cfg.block_size, rope_theta, device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scaled init for residual output projections, as in GPT-2/nanoGPT. Applied
        # identically to every run, so it cannot favour one configuration.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, QuantLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, dtype=torch.bfloat16
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")

        w = self.embed.quantized_weight()          # quantized every step (SPEC §2.3)
        x = F.embedding(idx, w).to(dtype)          # activations in bf16 (SPEC §2.4)

        cos, sin = self.rope_cos.to(x.device), self.rope_sin.to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_final(x)

        logits = F.linear(x, w.to(x.dtype), None)  # tied head, same quantized matrix
        if targets is None:
            return logits, None
        # Cross-entropy in fp32: bf16 logsumexp over the vocabulary loses real accuracy.
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.reshape(-1)
        )
        return logits, loss

    # ----------------------------------------------------------------------------------
    # Introspection used by training guards and the results row
    # ----------------------------------------------------------------------------------

    def quantized_layers(self) -> list[tuple[str, QuantLinear]]:
        return [(n, m) for n, m in self.named_modules() if isinstance(m, QuantLinear)]

    def check_no_collapse(self, min_nonzero_frac: float = 0.05) -> None:
        """Crash with the layer name if any quantized tensor collapsed (SPEC §3)."""
        if self.bits == BF16_CONTROL_BITS:
            return
        for name, layer in self.quantized_layers():
            layer.check_not_collapsed(name, min_nonzero_frac)

    def counted_params(self) -> int:
        """Quantized parameters actually instantiated. Must equal `count_params(cfg)`."""
        return sum(m.weight.numel() for _, m in self.quantized_layers())

    def counted_norm_params(self) -> int:
        """RMSNorm gains actually instantiated. Must equal `norm_params(cfg)`."""
        return sum(m.weight.numel() for m in self.modules() if isinstance(m, RMSNorm))

    def assert_matches_accounting(self) -> None:
        """The model and the accounting must agree exactly, or the x-axis is wrong."""
        actual, expected = self.counted_params(), count_params(self.cfg)
        if actual != expected:
            raise AssertionError(
                f"model instantiates {actual:,} quantized params but accounting.py counts "
                f"{expected:,} for {self.cfg}. The experiment's x-axis is wrong."
            )
        actual_n, expected_n = self.counted_norm_params(), norm_params(self.cfg)
        if actual_n != expected_n:
            raise AssertionError(
                f"model has {actual_n:,} norm gains, accounting counts {expected_n:,}"
            )
        # Nothing may exist outside those two categories: an unaccounted parameter is
        # storage the budget never paid for.
        total = sum(p.numel() for p in self.parameters())
        if total != expected + expected_n:
            unaccounted = total - expected - expected_n
            raise AssertionError(
                f"{unaccounted:,} parameters are outside the accounting "
                "(neither a quantized weight nor an RMSNorm gain)"
            )

    def param_groups(self, weight_decay: float) -> list[dict]:
        """AdamW groups: decay the matrices, never the 1-D norm gains.

        Standard practice (GPT-2, nanoGPT): weight decay on norm gains pulls them toward
        zero, which is actively harmful. Applied identically to every run, so it cannot
        act as a per-configuration tuning knob (SPEC §2.8).
        """
        decay = [p for p in self.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.dim() < 2]
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
