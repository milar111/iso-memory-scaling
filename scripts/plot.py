#!/usr/bin/env python
"""Figures for SPEC §8.

Four figures, each making one claim:

1. **headline** — val loss against *actual* stored bits per weight (stored weight bits over
   parameters), one line per budget, a shaded band and error bars at +/-1 seed standard
   deviation, the best point per budget annotated (PREREGISTRATION.md §6.1: "Actual bits,
   never target bits"). The noise is the point: commitment 4 requires it beside every
   gap, so a reader can see for themselves whether the argmin is separated from its
   neighbours.
2. **iso_budget** — val loss against the *actual* bit count, coloured by `b`. If the search
   did its job the points at one budget form a vertical stripe, so this figure is the visual
   audit of the iso-memory constraint rather than a result.
3. **money** — the same data read two ways: against parameters (the literature's framing)
   and against storage (the question this experiment asks). They disagree, which is P3.
4. **selection** — the selection gain (final val loss minus `val_loss_best`) per cell,
   mean +/-1 seed sd, and, for illustration only, the validation curves of seed 0 of four
   cells at 256 Kbit. Seed 0 is used for every cell; an earlier version showed the ternary
   seed with the largest rise (seed 3), which overstated the typical curve.

Embedding-bound cells (`embedding_frac > 0.5`, PREREGISTRATION.md §6.5) are drawn hollow
everywhere they appear. On the finished sweep there are none: the largest share is the
256 Kbit, b=8 cell's, just under the threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Embeddable TrueType (Type 42) in the PDFs: matplotlib's default is Type 3, which venue
# checkers reject. The PDF creation date comes from SOURCE_DATE_EPOCH (the Makefile sets it
# to 0), so regenerated figures are byte-stable.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import numpy as np  # noqa: E402

from isomem.accounting import EMBEDDING_BOUND_THRESHOLD  # noqa: E402
from isomem.analysis import load_cells, null_verdict, selection_gains  # noqa: E402
from isomem.eval import load_baseline  # noqa: E402
from isomem.fixed import GRID  # noqa: E402

# The first three categorical slots of the dataviz reference palette (blue, orange, aqua).
# The previous matplotlib defaults (#1f77b4, #d62728, #2ca02c) failed the palette validator's
# colour-vision check: red vs green measured Delta E 3.9 under deuteranopia, against a floor
# of 8. These pass every check in light mode (validate_palette.js --mode light: worst CVD
# Delta E 9.2 deutan, worst normal vision 27.6 adjacent / 24.0 with --pairs all); aqua sits at
# 2.74:1 against the surface, below 3:1, so every figure keeps a legend or direct labels and
# the tables carry the same numbers.
BUDGET_COLOR = {"256Kbit": "#2a78d6", "828Kbit": "#eb6834", "4Mbit": "#1baf7a"}
# The active corpus' own floor, read rather than hardcoded: 3.1648 on TinyStories, and a
# different number on any other corpus, where TinyStories' value would be meaningless.
BIGRAM_FLOOR = (_b.val_loss if (_b := load_baseline()) else 3.1648)


# Figures carry no titles or editorial text: the paper's captions state what each shows.
YLABEL = "validation loss (nats/token)"


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def _bit_ticks(ax, cells):
    """One tick per precision actually present. A hardcoded list silently drops b=16."""
    ticks = sorted({c.bits for c in cells if getattr(c, "usable", True)})
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())


def _rows(cells, name):
    return sorted([c for c in cells if c.budget_name == name and c.usable],
                  key=lambda c: c.bits)


def actual_bits_per_weight(c) -> float:
    """Stored weight bits over parameters: what the cell actually spends per weight.

    PREREGISTRATION.md §6.1 puts the headline on ACTUAL bits, never the
    grid's target b. The two differ by the 16-bit norm gains and by the search's budget
    error, most at the smallest budget.
    """
    return c.actual_bits / c.params


def is_embedding_bound(c) -> bool:
    """PREREGISTRATION.md §6.5: embedding_frac > 0.5, the threshold isomem.accounting uses."""
    return c.embedding_frac > EMBEDDING_BOUND_THRESHOLD


def fig_headline(cells, out: Path, rob_cells=()) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    n_bound = 0
    for name in [b["name"] for b in GRID["budgets_bits"]]:
        row = _rows(cells, name)
        if not row:
            continue
        x = np.array([actual_bits_per_weight(c) for c in row])
        y = np.array([c.mean for c in row])
        e = np.array([c.std for c in row])
        colour = BUDGET_COLOR.get(name, "#555555")
        partial = any(c.n < 5 for c in row)
        ax.fill_between(x, y - e, y + e, color=colour, alpha=0.18, linewidth=0, zorder=2)
        ax.errorbar(x, y, yerr=e, color=colour, marker="o", markersize=5, capsize=3,
                    linewidth=1.6, linestyle="--" if partial else "-",
                    label=f"{name}{' (partial)' if partial else ''}", zorder=3)
        # embedding-bound cells drawn hollow: most of the budget is not transformer
        eb = [(actual_bits_per_weight(c), c.mean) for c in row if is_embedding_bound(c)]
        n_bound += len(eb)
        if eb:
            ax.scatter(*zip(*eb), s=110, facecolors="none", edgecolors=colour,
                       linewidths=1.6, zorder=4)
        best = min(row, key=lambda c: c.mean)
        v = null_verdict(cells, name)
        ax.annotate(f"b*={best.bits:g}" + ("  (null)" if v.is_null else ""),
                    (actual_bits_per_weight(best), best.mean), textcoords="offset points",
                    xytext=(6, -13), fontsize=8, color=colour)
    # The shape-robustness sweep (ASPECT rule, 3 seeds) as an overlay, so the reader sees
    # how far a change of depth/width moves each cell: hollow diamonds, dotted line.
    for name in [b["name"] for b in GRID["budgets_bits"]]:
        row = _rows(list(rob_cells), name)
        if not row:
            continue
        colour = BUDGET_COLOR.get(name, "#555555")
        ax.errorbar([actual_bits_per_weight(c) for c in row], [c.mean for c in row],
                    yerr=[c.std for c in row], color=colour, marker="D", markersize=4.5,
                    markerfacecolor="white", capsize=2, linewidth=1.0, linestyle=":",
                    label=f"{name}, ASPECT (n={row[0].n})", zorder=3)
    ax.axhline(BIGRAM_FLOOR, color="#999999", linewidth=1, linestyle=":")
    ax.text(1.5, BIGRAM_FLOOR + 0.02, "bigram floor", fontsize=7.5, color="#777777")
    _bit_ticks(ax, cells)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.06 * (hi - lo), hi)          # room for the b* label under the lowest curve
    _style(ax, "actual stored bits per weight (weight bits / parameters)", YLABEL)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_iso_budget(cells, out: Path) -> None:
    """The audit figure: iso-budget lines must actually be iso."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    cmap = plt.get_cmap("viridis")
    allbits = sorted({c.bits for c in cells})
    for c in cells:
        if not c.usable:
            continue
        colour = cmap(allbits.index(c.bits) / max(1, len(allbits) - 1))
        ax.errorbar(c.actual_bits, c.mean, yerr=c.std, marker="o", markersize=6,
                    color=colour, capsize=2, linewidth=1,
                    markerfacecolor="none" if is_embedding_bound(c) else colour)
    # Labels are stacked per budget with a minimum vertical gap, so cells that land at
    # nearly the same loss stay readable; a thin leader line ties each label to its point.
    lo_y = min(c.mean for c in cells if c.usable)
    hi_y = max(c.mean for c in cells if c.usable)
    gap = 0.045 * (hi_y - lo_y)
    for name in sorted({c.budget_name for c in cells if c.usable}):  # set order is per-process
        col = sorted([c for c in cells if c.usable and c.budget_name == name],
                     key=lambda c: c.mean)
        ys, prev = [], None
        for c in col:
            y = c.mean if prev is None else max(c.mean, prev + gap)
            ys.append(y)
            prev = y
        shift = (ys[-1] - col[-1].mean) / 2          # re-centre the stack on the points
        for c, y in zip(col, ys):
            colour = cmap(allbits.index(c.bits) / max(1, len(allbits) - 1))
            ax.annotate(f"b={c.bits:g}", xy=(c.actual_bits, c.mean),
                        xytext=(c.actual_bits * 1.18, y - shift), fontsize=7,
                        color=colour, va="center",
                        arrowprops=dict(arrowstyle="-", color=colour, linewidth=0.5,
                                        alpha=0.6))
    for b in GRID["budgets_bits"]:
        ax.axvline(b["bits"], color="#bbbbbb", linewidth=0.8, linestyle=":")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.04 * (hi - lo))
    ax.set_xscale("log")
    ax.set_xlim(GRID["budgets_bits"][0]["bits"] * 0.8, GRID["budgets_bits"][-1]["bits"] * 1.6)
    ax.set_xticks([b["bits"] for b in GRID["budgets_bits"]])
    ax.set_xticklabels([f"{b['name']}\n{b['bits']:,} bits" for b in GRID["budgets_bits"]])
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    _style(ax, "actual weight bits stored", YLABEL)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_money(cells, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), dpi=160, sharey=True)
    for name in [b["name"] for b in GRID["budgets_bits"]]:
        row = _rows(cells, name)
        if not row:
            continue
        colour = BUDGET_COLOR.get(name, "#555")
        axes[0].plot([c.params for c in row], [c.mean for c in row], marker="o",
                     markersize=5, color=colour, linewidth=1.5, label=name)
        axes[1].errorbar([actual_bits_per_weight(c) for c in row], [c.mean for c in row],
                         yerr=[c.std for c in row], marker="o", markersize=5,
                         color=colour, capsize=3, linewidth=1.5, label=name)
        for c in row:
            axes[0].annotate(f"{c.bits:g}", (c.params, c.mean),
                             textcoords="offset points", xytext=(5, 3), fontsize=7,
                             color=colour)
    # The matched-parameter comparison is a vertical slice: same parameter count, two
    # different b, and necessarily two different storage budgets. Drawing it makes the
    # confound visible instead of leaving it to the caption.
    slice_at = 1.4e5
    lo = min((c for c in _rows(cells, "256Kbit")),
             key=lambda c: abs(c.params - slice_at), default=None)
    hi = min((c for c in _rows(cells, "828Kbit")),
             key=lambda c: abs(c.params - slice_at), default=None)
    if lo and hi:
        axes[0].axvline(slice_at, color="#888888", linewidth=0.9, linestyle=":")
        axes[0].annotate("", xy=(slice_at, hi.mean), xytext=(slice_at, lo.mean),
                         arrowprops=dict(arrowstyle="<->", color="#888888", linewidth=0.9))
    axes[0].set_xscale("log")
    axes[0].set_xticks([3e4, 6e4, 1e5, 2e5, 5e5, 1e6, 2e6])
    axes[0].get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v/1e6:.0f}M" if v >= 1e6 else f"{v/1000:.0f}k"))
    axes[0].get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    _style(axes[0], "parameters", YLABEL)
    _bit_ticks(axes[1], cells)
    _style(axes[1], "actual stored bits per weight (weight bits / parameters)", "")
    axes[1].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_selection(histories: dict, out: Path) -> None:
    """Why val_loss_best and the final loss diverge by an amount that falls with b.

    The left panel is the claim, aggregated over all five seeds with their spread: a single
    seed's curve is too noisy to show it, and plotting one would misrepresent the effect.
    The right panel is an illustration only: seed 0 of four 256 Kbit cells. Seed 0 is used
    for every cell; an earlier version showed the ternary seed with the largest rise (seed
    3), which overstated the typical curve. Seed 0 has the smallest ternary gain of the five
    seeds, so this panel understates it (``selGainTernSeedZero`` against ``selGainTernMean``
    in paper/generated/numbers.tex).
    """
    gains = selection_gains(histories)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), dpi=160)

    for name in [b["name"] for b in GRID["budgets_bits"]]:
        row = [g for g in gains if g.budget_name == name]
        if not row:
            continue
        colour = BUDGET_COLOR.get(name, "#555")
        partial = any(g.n < 5 for g in row)
        axes[0].errorbar([g.bits for g in row], [g.mean for g in row],
                         yerr=[0 if np.isnan(g.std) else g.std for g in row],
                         marker="o", markersize=5, capsize=3, color=colour,
                         linewidth=1.5, linestyle="--" if partial else "-",
                         label=f"{name}{' (partial)' if partial else ''}")
    axes[0].axhline(0, color="#bbbbbb", linewidth=0.8)
    _bit_ticks(axes[0], gains)
    _style(axes[0], "bits per weight (b)", "final val loss − minimum (nats/token)")
    axes[0].legend(frameon=False, fontsize=8.5)

    cmap = plt.get_cmap("viridis")
    seed = 0
    shown = [(f"B262144-b{b:g}-s{seed}", b) for b in (1.58, 3.0, 6.0, 8.0)]
    allb = [b for _, b in shown]
    for rid, b in shown:
        h = histories.get(rid)
        if not h:
            raise KeyError(f"{rid} has no validation history; the figure would drop a curve")
        steps = [e["step"] for e in h["history"]]
        val = [e["val"] for e in h["history"]]
        colour = cmap(allb.index(b) / max(1, len(allb) - 1))
        axes[1].plot(steps, val, color=colour, linewidth=1.2, label=f"b={b:g}")
        i = int(np.argmin(val))
        axes[1].scatter([steps[i]], [val[i]], color=colour, s=30, zorder=4)
    _style(axes[1], "step", YLABEL)
    axes[1].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    ap.add_argument("--histories", type=Path, default=None,
                    help="JSON of per-run validation histories, for the selection figure")
    ap.add_argument("--format", choices=["png", "pdf"], default="png",
                    help="file format of the figures (pdf for the paper)")
    args = ap.parse_args()
    ext = args.format
    args.out.mkdir(parents=True, exist_ok=True)

    cells = load_cells()
    fig_headline(cells, args.out / f"headline.{ext}", load_cells(sweep="robustness"))
    fig_iso_budget(cells, args.out / f"iso_budget.{ext}")
    fig_money(cells, args.out / f"money.{ext}")
    written = [f"headline.{ext}", f"iso_budget.{ext}", f"money.{ext}"]

    if args.histories and args.histories.exists():
        fig_selection(json.loads(args.histories.read_text()),
                      args.out / f"selection.{ext}")
        written.append(f"selection.{ext}")

    for name in written:
        print(f"  wrote {args.out / name}")


if __name__ == "__main__":
    main()
