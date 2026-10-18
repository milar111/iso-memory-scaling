"""Architecture search under the bit constraint (SPEC §1.3).

Given a budget `B` and a bit-width `b`, choose `(n_layer, n_embd, mlp_ratio)` so that
`count_weight_bits` lands within tolerance of `B`. `n_head` is derived, never chosen.

The tie-break is a methodological choice, not a detail
-----------------------------------------------------
SPEC §1.3 says only "pick the one minimising |actual_bits - B|". That rule usually does not
identify a single configuration: many shapes hit the same budget to within a fraction of a
percent, and they are not equivalent models — a deep-narrow and a shallow-wide network at
the same parameter count train differently.

So the tie-break is a free parameter sitting directly on the experiment's axis, and an
undocumented one is exactly how a sweep quietly manufactures its conclusion (SPEC §2.8).
It is therefore named, fixed, and applied identically to every cell:

``TIGHTEST``    (default) the spec-literal reading: minimise |error|, break ties
                lexicographically by (n_layer, n_embd, mlp_ratio). Budget accuracy is
                maximised and shape falls where the constraint puts it.

``ASPECT``      among every configuration within the full ±tol band, choose the one whose
                aspect ratio ``n_embd / n_layer`` is closest to ``TARGET_ASPECT``, then
                minimise error. Trades budget accuracy for shape stability.

Shape is mostly NOT free, and that is a property of the question
---------------------------------------------------------------
Measured over the search space, the aspect ratios reachable within ±5% at B = 256 Kbit are
*disjoint* across bit-widths: b=3 can only reach 4.0–5.3, b=4 only 5.3–8.0, b=6 only 10.7,
b=8 only 16.0. No tie-break can equalise them. Holding a fixed weight budget while varying
precision necessarily walks the model from a deep-narrow shape to a shallow-wide one.

So at the smaller budgets, precision and shape are the same axis, and the honest move is to
say so and measure it rather than to pretend a search rule disentangles them. At B = 4 Mbit
there is real freedom (aspect 4 to 192), so ``ASPECT`` is available there as a robustness
check: rerunning a budget under both rules bounds how much of any observed effect is shape
rather than precision.

`aspect` is logged per run so a reviewer can see it on every plot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Literal

from isomem.accounting import (
    ModelConfig,
    count_weight_bits,
    embedding_frac,
    is_embedding_bound,
)
from isomem.fixed import FIXED, GRID

ShapeRule = Literal["ASPECT", "TIGHTEST"]

# n_embd / n_layer. 10 minimises the worst-case distance from target over all 20 feasible
# cells (worst gap 6.8, worst budget error 4.15%); every other target tested is worse on
# both counts. Used only by the ASPECT rule.
TARGET_ASPECT = 10.0


class InfeasibleBudget(Exception):
    """No configuration can reach this budget. Carries the arithmetic (SPEC §3)."""


@dataclass(frozen=True)
class SearchResult:
    config: ModelConfig
    actual_bits: int
    target_bits: int
    bits_per_weight: float

    @property
    def rel_error(self) -> float:
        return (self.actual_bits - self.target_bits) / self.target_bits

    @property
    def embedding_frac(self) -> float:
        return embedding_frac(self.config, self.bits_per_weight)

    @property
    def embedding_bound(self) -> bool:
        return is_embedding_bound(self.config, self.bits_per_weight)

    @property
    def aspect(self) -> float:
        return self.config.n_embd / self.config.n_layer


def _search_space(vocab: int, block_size: int) -> Iterator[ModelConfig]:
    """All valid shapes, in a fixed order so the search is deterministic (SPEC §6)."""
    s = GRID["search"]
    lo, hi = s["n_layer_range"]
    for n_layer in range(lo, hi + 1):
        for n_embd in range(s["n_embd_min"], s["n_embd_max"] + 1, s["n_embd_step"]):
            for mlp_ratio in s["mlp_ratios"]:
                try:
                    yield ModelConfig(n_layer, n_embd, float(mlp_ratio), vocab, block_size)
                except ValueError:
                    continue  # rejected by the constraints in ModelConfig


def smallest_possible_bits(bits: float, vocab: int, block_size: int) -> tuple[ModelConfig, int]:
    """The floor of the search space: no configuration can be smaller than this."""
    best: tuple[ModelConfig, int] | None = None
    for cfg in _search_space(vocab, block_size):
        n = count_weight_bits(cfg, bits)
        if best is None or n < best[1]:
            best = (cfg, n)
    assert best is not None
    return best


def _infeasibility_report(
    budget_bits: int, bits: float, vocab: int, block_size: int
) -> str:
    """Diagnose *why* a cell is unreachable, with the arithmetic (SPEC §3).

    A generic "no config within tolerance" would hide the interesting case: at small
    budgets the embedding table alone can exceed the entire budget, which is SPEC §2.10's
    embedding-bound regime taken to its limit and a finding in its own right.
    """
    s = GRID["search"]
    min_embd = s["n_embd_min"]
    emb_bits = math.ceil(vocab * min_embd * bits)
    floor_cfg, floor_bits = smallest_possible_bits(bits, vocab, block_size)
    lines = [
        f"no configuration reaches B={budget_bits:,} bits at b={bits}",
        f"  smallest legal shape: n_layer={floor_cfg.n_layer}, n_embd={floor_cfg.n_embd}, "
        f"mlp_ratio={floor_cfg.mlp_ratio} -> {floor_bits:,} bits "
        f"({floor_bits / budget_bits:.1%} of budget)",
    ]
    if emb_bits >= budget_bits:
        lines.append(
            f"  CAUSE: the tied embedding table alone costs vocab*n_embd_min*b = "
            f"{vocab}*{min_embd}*{bits} = {emb_bits:,} bits = "
            f"{emb_bits / budget_bits:.1%} of the budget, leaving nothing for any block."
        )
        lines.append(
            "  Not fixable: vocab and head_dim are held fixed across all runs (SPEC §1.2), "
            "and changing the vocabulary makes perplexity incomparable."
        )
        lines.append(
            "  This is the embedding-bound regime (SPEC §2.10) at its limit. Report it as "
            "a result, not an omission."
        )
    else:
        lines.append("  CAUSE: the search space floor exceeds the budget.")
    return "\n".join(lines)


def find_config(
    budget_bits: int,
    bits: float,
    vocab: int,
    tol: float | None = None,
    block_size: int | None = None,
    shape_rule: ShapeRule = "TIGHTEST",
) -> tuple[ModelConfig, int]:
    """Return the configuration and its ACTUAL bits. Raises if none is within `tol`.

    Deterministic: the same inputs return the same configuration, always.
    """
    tol = GRID["search"]["tolerance"] if tol is None else tol
    block_size = FIXED["model"]["block_size"] if block_size is None else block_size

    candidates: list[tuple[float, ModelConfig, int]] = []
    for cfg in _search_space(vocab, block_size):
        actual = count_weight_bits(cfg, bits)
        candidates.append((abs(actual - budget_bits) / budget_bits, cfg, actual))
    if not candidates:
        raise InfeasibleBudget(f"empty search space for vocab={vocab}")

    best_err = min(err for err, _, _ in candidates)
    if best_err > tol:
        raise InfeasibleBudget(
            _infeasibility_report(budget_bits, bits, vocab, block_size)
            + f"\n  best achievable error {best_err:.2%} exceeds tolerance {tol:.0%}"
        )

    if shape_rule == "TIGHTEST":
        # Spec-literal: minimise error, break ties lexicographically.
        err, cfg, actual = min(
            candidates,
            key=lambda c: (c[0], c[1].n_layer, c[1].n_embd, c[1].mlp_ratio),
        )
        return cfg, actual

    # ASPECT: among everything inside the tolerance band, hold shape as steady as the
    # constraint allows, then minimise error.
    band = [c for c in candidates if c[0] <= tol]
    err, cfg, actual = min(
        band,
        key=lambda c: (
            abs(c[1].n_embd / c[1].n_layer - TARGET_ASPECT),
            c[0],
            c[1].n_layer,
            c[1].n_embd,
            c[1].mlp_ratio,
        ),
    )
    return cfg, actual


def search_cell(
    budget_bits: int, bits: float, vocab: int | None = None, **kw
) -> SearchResult:
    vocab = FIXED["tokenizer"]["vocab_size"] if vocab is None else vocab
    cfg, actual = find_config(budget_bits, bits, vocab, **kw)
    return SearchResult(cfg, actual, budget_bits, bits)


def grid_cells() -> list[tuple[str, int, float]]:
    """Every (budget_name, budget_bits, bits) cell in the experiment."""
    return [
        (b["name"], b["bits"], float(bits))
        for b in GRID["budgets_bits"]
        for bits in GRID["bits_per_weight"]
    ]


def search_grid(
    vocab: int | None = None, **kw
) -> tuple[list[tuple[str, SearchResult]], list[tuple[str, float, int, str]]]:
    """Search every cell. Returns (feasible results, infeasible cells with diagnoses)."""
    feasible: list[tuple[str, SearchResult]] = []
    infeasible: list[tuple[str, float, int, str]] = []
    for name, budget, bits in grid_cells():
        try:
            feasible.append((name, search_cell(budget, bits, vocab, **kw)))
        except InfeasibleBudget as exc:
            infeasible.append((name, bits, budget, str(exc)))
    return feasible, infeasible
