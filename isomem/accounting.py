"""Bit accounting (SPEC §2.1).

Every downstream number depends on this module being right. The counting rules here are
the *definition* of the experiment's x-axis, so `model.py` must conform to this file, not
the other way round. A test (not part of this release) checks two
hand-computed configurations with no tolerance at all.

What is counted as weight storage
---------------------------------
Per transformer block, with ``d = n_embd`` and ``r = mlp_ratio``:

* attention QKV projection      3 * d * d      (one fused Linear, no bias)
* attention output projection   1 * d * d
* MLP up projection             r * d * d
* MLP down projection           r * d * d

giving ``(4 + 2r) * d**2`` per block, exactly the formula in SPEC §1.3.

Plus, once for the whole model:

* the tied embedding / output head matrix, ``vocab * d`` (SPEC §1.2: weight tying, counted once)

These are the *quantized* tensors: they are stored at ``bits_per_weight`` bits each.

Counted separately, at their real precision
-------------------------------------------
RMSNorm gains are kept in bf16 and counted anyway (SPEC §1.2, §2.1) even though they are
negligible. There are ``2 * n_layer + 1`` of them, each of width ``d``: two per block
(pre-attention, pre-MLP) and one final norm before the head.

Excluded from the budget but logged (SPEC §2.1)
-----------------------------------------------
KV cache, peak activations, optimizer state. The KV-cache exclusion is a deliberate scope
decision and a stated limitation: at a fixed weight budget, wider models carry larger KV
caches, so a full hardware accounting would penalise them. ``kv_bits_at_full_context``
exists so a reviewer can see the magnitude.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# RMSNorm gains are held in bf16 (SPEC §1.2 `norm_dtype`).
NORM_DTYPE_BITS = 16


class BitBudgetError(AssertionError):
    """Raised when a configuration misses its bit budget. Crash, never warn (SPEC §2.1)."""


@dataclass(frozen=True)
class ModelConfig:
    """A point in the architecture search space.

    Frozen so a config cannot be mutated after its bits have been asserted.
    ``n_head`` is derived, never chosen: SPEC §1.3 fixes ``n_head = n_embd // 32``.
    """

    n_layer: int
    n_embd: int
    mlp_ratio: float
    vocab_size: int
    block_size: int
    head_dim: int = 32

    def __post_init__(self) -> None:
        # Search-space constraints from SPEC §1.3, asserted at construction so an invalid
        # config cannot exist anywhere in the program.
        if self.n_embd % self.head_dim != 0:
            raise ValueError(f"n_embd={self.n_embd} not divisible by head_dim={self.head_dim}")
        if self.n_embd < self.head_dim:
            raise ValueError(f"n_embd={self.n_embd} < head_dim={self.head_dim}")
        if self.n_layer < 2:
            raise ValueError(f"n_layer={self.n_layer} < 2")
        if self.n_head < 1:
            raise ValueError(f"n_head={self.n_head} < 1")
        if self.vocab_size < 1 or self.block_size < 1:
            raise ValueError("vocab_size and block_size must be positive")
        # mlp_ratio is a half-integer grid value; d**2 * r must be a whole number of params.
        if (self.mlp_ratio * self.n_embd * self.n_embd) % 1 != 0:
            raise ValueError(f"mlp_ratio={self.mlp_ratio} gives fractional MLP params")

    @property
    def n_head(self) -> int:
        return self.n_embd // self.head_dim

    @property
    def mlp_hidden(self) -> int:
        hidden = self.mlp_ratio * self.n_embd
        if hidden % 1 != 0:
            raise ValueError(f"mlp_hidden={hidden} is not an integer")
        return int(hidden)


# --------------------------------------------------------------------------------------
# Parameter counts
# --------------------------------------------------------------------------------------


def block_params(cfg: ModelConfig) -> int:
    """Quantized parameters in ONE transformer block: (4 + 2 * mlp_ratio) * n_embd**2."""
    d = cfg.n_embd
    attn = 3 * d * d + d * d          # fused QKV, then output projection
    mlp = 2 * cfg.mlp_hidden * d      # up then down
    return attn + mlp


def embedding_params(cfg: ModelConfig) -> int:
    """The tied embedding/head matrix. Counted exactly once (SPEC §1.2)."""
    return cfg.vocab_size * cfg.n_embd


def count_params(cfg: ModelConfig) -> int:
    """Total *quantized* parameters. Matches SPEC §1.3 exactly:

    params = n_layer * (4 * n_embd**2 + 2 * mlp_ratio * n_embd**2) + vocab * n_embd

    RMSNorm gains are NOT included here; they are not quantized. They are counted in
    ``count_weight_bits`` at bf16, per SPEC §2.1.
    """
    return cfg.n_layer * block_params(cfg) + embedding_params(cfg)


def norm_params(cfg: ModelConfig) -> int:
    """RMSNorm gains: two per block plus one final norm, each of width n_embd."""
    return (2 * cfg.n_layer + 1) * cfg.n_embd


# --------------------------------------------------------------------------------------
# Bit counts
# --------------------------------------------------------------------------------------


def _quantized_bits(n_params: int, bits: float) -> int:
    """Bits needed to store ``n_params`` weights at ``bits`` bits per weight.

    Fractional bit-widths are real storage costs, not metaphors: ternary at log2(3) =
    1.585 bits/weight means 5 trits packed into an 8-bit byte. We round the *total* up to
    a whole number of bits, which is what any real packing achieves, rather than rounding
    per weight (which would silently inflate ternary by 26%).
    """
    if n_params < 0:
        raise ValueError("n_params must be non-negative")
    if bits <= 0:
        raise ValueError("bits must be positive")
    return math.ceil(n_params * bits)


def count_weight_bits(cfg: ModelConfig, bits: float) -> int:
    """Total weight storage in bits: quantized tensors + bf16 RMSNorm gains (SPEC §2.1)."""
    return _quantized_bits(count_params(cfg), bits) + norm_params(cfg) * NORM_DTYPE_BITS


def embedding_bits(cfg: ModelConfig, bits: float) -> int:
    """Bits held by the tied embedding/head matrix alone."""
    return _quantized_bits(embedding_params(cfg), bits)


def embedding_frac(cfg: ModelConfig, bits: float) -> float:
    """Share of the weight budget spent on the embedding table.

    SPEC §2.10: when this exceeds 0.5 the run is in the *embedding-bound regime*. That is
    a finding about real SRAM accelerators, not a bug, and such points are reported as a
    separate regime rather than dropped.
    """
    return embedding_bits(cfg, bits) / count_weight_bits(cfg, bits)


EMBEDDING_BOUND_THRESHOLD = 0.5


def is_embedding_bound(cfg: ModelConfig, bits: float) -> bool:
    return embedding_frac(cfg, bits) > EMBEDDING_BOUND_THRESHOLD


def kv_bits_at_full_context(cfg: ModelConfig, dtype_bits: int = 16) -> int:
    """KV cache bits for a single sequence at full context. EXCLUDED from the budget.

    Two tensors (K and V), each ``n_layer * block_size * n_embd`` elements, since
    ``n_head * head_dim == n_embd``. Logged per run so a reviewer can see how much a full
    hardware accounting would penalise the wider (higher-precision) configurations.
    """
    if dtype_bits <= 0:
        raise ValueError("dtype_bits must be positive")
    return 2 * cfg.n_layer * cfg.block_size * cfg.n_embd * dtype_bits


# --------------------------------------------------------------------------------------
# Budget enforcement
# --------------------------------------------------------------------------------------

DEFAULT_TOLERANCE = 0.05


def budget_error(cfg: ModelConfig, bits: float, budget_bits: int) -> float:
    """Signed relative error of actual bits against the target budget."""
    return (count_weight_bits(cfg, bits) - budget_bits) / budget_bits


def assert_within_budget(
    cfg: ModelConfig, bits: float, budget_bits: int, tol: float = DEFAULT_TOLERANCE
) -> int:
    """Crash unless actual weight bits are within ``tol`` of the budget (SPEC §2.1).

    Called before training starts. Do not warn — raise.
    """
    actual = count_weight_bits(cfg, bits)
    err = abs(actual - budget_bits) / budget_bits
    if err > tol:
        raise BitBudgetError(
            f"bit budget missed: actual={actual:,} target={budget_bits:,} "
            f"error={err:.4%} > tol={tol:.2%} for cfg={cfg} bits={bits}"
        )
    return actual


def accounting_row(cfg: ModelConfig, bits: float, budget_bits: int) -> dict[str, object]:
    """The accounting columns of a `results.csv` row (SPEC §5)."""
    actual = count_weight_bits(cfg, bits)
    return {
        "budget_bits_target": budget_bits,
        "budget_bits_actual": actual,
        "bits_per_weight": bits,
        "n_layer": cfg.n_layer,
        "n_embd": cfg.n_embd,
        "n_head": cfg.n_head,
        "mlp_ratio": cfg.mlp_ratio,
        "vocab": cfg.vocab_size,
        "params_counted": count_params(cfg),
        "norm_params": norm_params(cfg),
        "embedding_bits": embedding_bits(cfg, bits),
        "embedding_frac": embedding_frac(cfg, bits),
        "embedding_bound": is_embedding_bound(cfg, bits),
        "kv_bits_full_ctx": kv_bits_at_full_context(cfg),
        "budget_rel_error": budget_error(cfg, bits, budget_bits),
    }
