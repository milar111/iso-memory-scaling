"""Analysis of `results.csv`: aggregation, the null criterion, and significance (SPEC §8).

Two rules run through everything here.

**Derive, do not trust.** `undertrained` changed meaning partway through the sweep
(PREREGISTRATION.md Amendment A3 moved convergence from a tail-drift test to the token
budget), so rows written before and after carry different semantics. `tokens_seen` is
correct in every row and the budget is a deterministic function of `params_counted`, so the
flag is recomputed here rather than read. Any column whose meaning could drift gets the
same treatment.

**One row per run.** `results.csv` is append-only by design, so a run can appear more than
once: an infeasible cell was re-recorded on every tick before that defect was fixed, and a
run that crashed on the schema rename was retried and then succeeded. Both are honest
history and both stay in the published file, so the loader keeps only the last row per
`run_id` instead of averaging a run with itself.

**Report the noise next to the number.** PREREGISTRATION.md §4 defines the null criterion in
terms of the seed standard deviation, and commitment 4 requires that std to be printed
beside every claimed gap. `summarise()` therefore returns the std alongside every mean, and
`null_verdict()` returns the arithmetic it used, not just a boolean.
"""

from __future__ import annotations

import csv
import io
import math
import statistics as st
from dataclasses import dataclass
from pathlib import Path

from isomem.corpus import ACTIVE as CORPUS
from isomem.fixed import GRID
from isomem.sweep import token_budget

RESULTS_CSV = CORPUS.results_csv

# PREREGISTRATION.md §4: the secondary Welch test is two-sided at this level.
ALPHA = 0.05

# PREREGISTRATION.md §4: at n=5 a two-sample Welch test has ~NOMINAL_POWER power against a
# gap of about POWER_SD_MULTIPLE standard deviations. Gaps smaller than that are reported as
# undetectable. Both are the preregistration's stated values, not a measured power.
NOMINAL_POWER = 0.80
POWER_SD_MULTIPLE = 2.0


@dataclass(frozen=True)
class Cell:
    """One (budget, bits) point: its seeds, aggregated."""

    budget_name: str
    budget_bits: int
    bits: float
    sweep: str
    shape_rule: str
    n_layer: int
    n_embd: int
    mlp_ratio: float
    params: int
    actual_bits: int
    embedding_frac: float
    embedding_bound: bool
    aspect: float
    val_losses: list[float]
    tokens: list[int]
    undertrained: list[bool]

    @property
    def n(self) -> int:
        return len(self.val_losses)

    @property
    def mean(self) -> float:
        return st.mean(self.val_losses)

    @property
    def std(self) -> float:
        """Sample std over seeds. The quantity the null criterion is defined on."""
        return st.stdev(self.val_losses) if self.n > 1 else float("nan")

    @property
    def ppl(self) -> float:
        return math.exp(self.mean)

    @property
    def usable(self) -> bool:
        """Every seed trained to its full budget (SPEC §2.5)."""
        return self.n > 0 and not any(self.undertrained)

    @property
    def min_detectable_gap(self) -> float:
        return POWER_SD_MULTIPLE * self.std


def load_rows(path: Path | None = None, dedupe: bool = True) -> list[dict]:
    """Rows from `results.csv`, last-write-wins per `run_id` (see module docstring).

    Pass ``dedupe=False`` to see the raw file, which is what the published artifact holds.
    """
    path = path or RESULTS_CSV
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not dedupe:
        return rows
    last: dict[str, dict] = {}
    for r in rows:
        last[r["run_id"]] = r
    return list(last.values())


# The columns scripts/test_read.py fills, once, in the complete primary rows of results.csv
# (PREREGISTRATION.md §14). Everything frozen from results.csv before that read -- the A7
# plan.json's hash, the committed copies paper_numbers.py checks as prefixes -- is checked
# against results_csv_before_test_read(current file), so that read, and only it, passes.
TEST_READ_COLUMNS = ("test_loss", "test_ppl")


def results_csv_before_test_read(data: bytes) -> bytes:
    """results.csv as it stood before the §14 test read: the TEST_READ_COLUMNS cells blanked
    in exactly the rows test_read.py writes (status complete, sweep primary), serialised the
    way test_read.py writes the file. Every other byte is kept: a test value in any other
    row, or any other changed cell, still shows. A file without those columns, or not in
    test_read.py's canonical csv form, is returned unchanged -- never normalised."""
    text = data.decode("utf-8")
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows or not all(c in rows[0] for c in TEST_READ_COLUMNS):
        return data
    if _canonical(rows) != text:
        return data
    head = rows[0]
    idx = [head.index(c) for c in TEST_READ_COLUMNS]
    st_i, sw_i = head.index("status"), head.index("sweep")
    blanked = [head]
    for r in rows[1:]:
        r = list(r)
        if len(r) == len(head) and r[st_i] == "complete" and r[sw_i] == "primary":
            for i in idx:
                r[i] = ""
        blanked.append(r)
    return _canonical(blanked).encode("utf-8")


def _canonical(rows: list[list[str]]) -> str:
    out = io.StringIO(newline="")
    csv.writer(out).writerows(rows)
    return out.getvalue()


def duplicate_run_ids(path: Path | None = None) -> dict[str, list[str]]:
    """Run ids appearing more than once, with the statuses seen. For the write-up."""
    seen: dict[str, list[str]] = {}
    for r in load_rows(path, dedupe=False):
        seen.setdefault(r["run_id"], []).append(r["status"])
    return {k: v for k, v in seen.items() if len(v) > 1}


def is_undertrained(row: dict) -> bool:
    """Recomputed, never read from the column (see module docstring)."""
    want = token_budget(int(row["params_counted"]))
    return int(row["tokens_seen"]) < 0.999 * want


def load_cells(path: Path | None = None, sweep: str = "primary") -> list[Cell]:
    """Aggregate completed runs into cells. Deterministic order: budget, then bits."""
    groups: dict[tuple, list[dict]] = {}
    for r in load_rows(path):
        if r["status"] != "complete" or r["sweep"] != sweep:
            continue
        groups.setdefault((r["budget_name"], float(r["bits_per_weight"])), []).append(r)

    budget_order = {b["name"]: i for i, b in enumerate(GRID["budgets_bits"])}
    cells = []
    for (name, bits), rs in groups.items():
        f = rs[0]
        cells.append(Cell(
            budget_name=name, budget_bits=int(f["budget_bits_target"]), bits=bits,
            sweep=f["sweep"], shape_rule=f["shape_rule"],
            n_layer=int(f["n_layer"]), n_embd=int(f["n_embd"]),
            mlp_ratio=float(f["mlp_ratio"]), params=int(f["params_counted"]),
            actual_bits=int(f["budget_bits_actual"]),
            embedding_frac=float(f["embedding_frac"]),
            embedding_bound=f["embedding_bound"] == "True",
            aspect=float(f["aspect"]),
            val_losses=[float(r["val_loss_best"]) for r in rs],
            tokens=[int(r["tokens_seen"]) for r in rs],
            undertrained=[is_undertrained(r) for r in rs],
        ))
    return sorted(cells, key=lambda c: (budget_order.get(c.budget_name, 99), c.bits))


# --------------------------------------------------------------------------------------
# The null criterion (PREREGISTRATION.md §4)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NullVerdict:
    budget_name: str
    n_cells: int
    best_bits: float
    best_mean: float
    gap: float
    s_max: float
    is_null: bool
    adjacent: list[float]
    note: str = ""

    def describe(self) -> str:
        if self.note:
            return f"{self.budget_name}: {self.note}"
        verdict = "NULL" if self.is_null else "not null"
        return (
            f"{self.budget_name}: argmin at b={self.best_bits:g} "
            f"(val {self.best_mean:.4f}); gap to nearest adjacent {self.gap:.4f}, "
            f"max seed std over the triple {self.s_max:.4f} -> {verdict}"
        )


def null_verdict(cells: list[Cell], budget_name: str) -> NullVerdict:
    """NULL at this budget if `gap <= s_max` (PREREGISTRATION.md §4).

    `gap` is the smallest distance in mean val loss from the argmin to a bit-width adjacent
    to it in the grid; `s_max` is the largest seed std over the argmin and those neighbours.
    Only cells where every seed trained to its full budget are eligible.
    """
    row = [c for c in cells if c.budget_name == budget_name and c.usable]
    if len(row) < 2:
        return NullVerdict(budget_name, len(row), float("nan"), float("nan"),
                           float("nan"), float("nan"), True, [],
                           note=f"only {len(row)} usable cells; cannot judge")

    row.sort(key=lambda c: c.bits)
    best = min(row, key=lambda c: c.mean)
    i = row.index(best)
    neighbours = [row[j] for j in (i - 1, i + 1) if 0 <= j < len(row)]

    gap = min(abs(n.mean - best.mean) for n in neighbours)
    s_max = max(c.std for c in [best, *neighbours] if not math.isnan(c.std))
    return NullVerdict(budget_name, len(row), best.bits, best.mean, gap, s_max,
                       gap <= s_max, [n.bits for n in neighbours])


def welch(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Welch's t statistic, degrees of freedom, and two-sided p (SPEC §4 secondary).

    p is computed from the t distribution via the regularised incomplete beta function, so
    the module has no SciPy dependency.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan"), float("nan")
    va, vb = st.variance(a) / na, st.variance(b) / nb
    if va + vb == 0:
        return float("inf"), float("nan"), 0.0
    t = (st.mean(a) - st.mean(b)) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
    x = df / (df + t * t)
    return t, df, _betainc_half(df / 2, x)


def _betainc_half(a: float, x: float, terms: int = 200) -> float:
    """I_x(a, 1/2), which is the two-sided p-value for |t| with `2a` degrees of freedom."""
    b = 0.5
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # continued fraction for the regularised incomplete beta
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(terms):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return min(1.0, max(0.0, front * (f - 1.0)))


def pairwise_tests(cells: list[Cell], budget_name: str) -> list[dict]:
    """Welch tests between the argmin and every other bit-width at this budget."""
    row = sorted([c for c in cells if c.budget_name == budget_name and c.usable],
                 key=lambda c: c.bits)
    if len(row) < 2:
        return []
    best = min(row, key=lambda c: c.mean)
    out = []
    for c in row:
        if c is best:
            continue
        t, df, p = welch(best.val_losses, c.val_losses)
        gap = c.mean - best.mean
        out.append({
            "budget": budget_name, "best_bits": best.bits, "other_bits": c.bits,
            "gap": gap, "pooled_sd": max(best.std, c.std), "t": t, "df": df, "p": p,
            "significant": p < ALPHA,
            "detectable": abs(gap) >= POWER_SD_MULTIPLE * max(best.std, c.std),
        })
    return out


# --------------------------------------------------------------------------------------
# Selection gain: how much of `val_loss_best` is the minimum rather than the endpoint
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionGain:
    """How far a run's validation curve rose after the minimum it reports.

    The preregistered primary metric is `val_loss_best`, a minimum over validation
    evaluations. Low-`b` cells run about 1.5x more steps for the same bit budget, so they
    take more draws at that minimum, and a minimum over more draws is biased downward. The
    bias would land directly on the axis being measured, so it is quantified rather than
    assumed negligible.

    What the gain measures is the rise after the minimum, not the downward bias of the
    minimum itself; the two are not separated here. Whether the rise is late overfitting or
    evaluation noise, and how large the selection bias is at each `b`, has not been measured.
    (An earlier version of this docstring asserted both, from an uncomputed comparison.)
    """

    budget_name: str
    bits: float
    n: int
    gains: list[float]          # final val loss minus the minimum, per seed

    @property
    def mean(self) -> float:
        return st.mean(self.gains)

    @property
    def std(self) -> float:
        return st.stdev(self.gains) if self.n > 1 else float("nan")


def selection_gains(histories: dict, path: Path | None = None,
                    sweep: str = "primary") -> list[SelectionGain]:
    """Per-cell selection gain, from `{run_id: {"history": [{"step","train","val"}, ...]}}`."""
    groups: dict[tuple, list[float]] = {}
    for r in load_rows(path):
        if r["status"] != "complete" or r["sweep"] != sweep:
            continue
        entry = histories.get(r["run_id"])
        if not entry or len(entry.get("history", [])) < 20:
            continue
        v = [e["val"] for e in entry["history"]]
        groups.setdefault((r["budget_name"], float(r["bits_per_weight"])), []).append(
            v[-1] - min(v)
        )
    budget_order = {b["name"]: i for i, b in enumerate(GRID["budgets_bits"])}
    out = [SelectionGain(name, bits, len(g), g) for (name, bits), g in groups.items()]
    return sorted(out, key=lambda s: (budget_order.get(s.budget_name, 99), s.bits))


def selection_gain_is_monotonic(gains: list[SelectionGain], budget_name: str
                                ) -> tuple[bool, list[str]]:
    """Does the gain fall as `b` rises, and are any inversions inside seed noise?"""
    row = sorted([g for g in gains if g.budget_name == budget_name], key=lambda g: g.bits)
    inversions = []
    for a, b in zip(row, row[1:]):
        if b.mean > a.mean:
            noise = max(a.std, b.std)
            inside = "within" if (b.mean - a.mean) <= noise else "OUTSIDE"
            inversions.append(
                f"b={a.bits:g}->{b.bits:g}: +{b.mean - a.mean:.4f}, {inside} seed sd {noise:.4f}"
            )
    return not inversions, inversions
