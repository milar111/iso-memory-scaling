#!/usr/bin/env python
"""Which weight precision does the fitted law of Kumar et al. favour at fixed weight memory?

Kumar et al. (arXiv:2411.04330, "Scaling Laws for Precision") fit, for quantization-aware
training with the weight precision ``P`` varied and the activations and the cache held at
high precision, the loss

    L(N, D) = A [N (1 - exp(-P / gamma_w))]^(-alpha) + B D^(-beta) + E        (their Eq. 3)

with ``alpha = beta`` and the constants of their Table 2 (Appendix K): A = 4.299e3,
alpha = 0.4965, B = 1.806e4, E = 2.7648, gamma_w = 2.6745, n_w = 0.3037. Appendix K does not
say which of their fits Table 2 belongs to (Figures 3 and 4 fit Chinchilla with N replaced by
N_eff(P_w) alone, Figures 5 to 7 use the full law), so attributing Table 2's gamma_w and n_w
to the weights-only Eq. 3 is my assumption; the ranking below depends only on gamma_w and n_w.
Appendix K adds
that a fit of the form ``exp(x / gamma)`` in the main text "is really computed with offset
``exp(x / gamma + n)``"; the sign of the offset is not stated. Appendix N reports
gamma_w = 2.8111 and an exponent bias b = 0.1240 for a floating-point fit; that b plays the
role of the offset n is not stated, and the two floating-point rankings with an offset
assume it (my assumption).

Kumar et al. do not report which precision minimises the loss at a fixed *weight memory*;
they state a compute-optimal precision of about 7 to 8 bits (their Sec. 4.3.2). This script
is my own evaluation of their published law and constants, nothing they report. At fixed
weight memory ``M`` (bits) and fixed ``D``, ``N = M / P``, so the loss is a decreasing
function of ``f(P) / P`` where ``N_eff = N f(P)``: the precision with the largest
``f(P) / P`` has the lowest predicted loss.

Because the sign of the offset is not stated, three readings of ``f`` are evaluated:

* ``main_text``: f(P) = 1 - exp(-P / gamma_w) (Eq. 3 as printed, no offset);
* ``offset_minus``: f(P) = 1 - exp(-(P / gamma_w - n_w));
* ``offset_plus``: f(P) = 1 - exp(-(P / gamma_w + n_w)).

What holds under all three, and is therefore independent of that reading, is the ranking
inside the range their sweep covers (weight precisions 3 to 12 bits, their Sec. 4):
3 > 4 > 5 > 6 > 7 > 8 > 9 > 10 > 11 > 12. Over the
grid of this paper the readings disagree on 1.58 and 2 bits, and a continuous maximum
inside the fitted range exists under none of them (``offset_minus`` has one at about 2.66
bits, below the range; the other two decrease). The assertions below check these
statements; the output is ``results/kumar_derivation.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SOURCE = {
    "paper": "Kumar et al., Scaling Laws for Precision, arXiv:2411.04330",
    "law": "Eq. 3: L(N, D) = A [N (1 - exp(-P_w / gamma_w))]^(-alpha) + B D^(-beta) + E",
    "constants": "Table 2 (Appendix K); Appendix K does not state which fit Table 2 belongs to, "
                 "so using its gamma_w and n_w in Eq. 3 is an assumption; the offset form "
                 "exp(x / gamma + n) is stated in Appendix K",
    "floating_point_constants": "Appendix N: gamma_w = 2.8111, exponent bias b = 0.1240; "
                                "Appendix N does not say that b plays the role of the offset n "
                                "of the integer-type form, so the two floating-point rankings "
                                "with an offset use that identification as my assumption",
}
A, ALPHA, B, E = 4.299e3, 0.4965, 1.806e4, 2.7648      # Table 2; alpha = beta (Appendix K)
GAMMA_W, N_W = 2.6745, 0.3037                          # Table 2
GAMMA_W_FP, N_W_FP = 2.8111, 0.1240                    # Appendix N (floating-point fit)

READINGS = ("main_text", "offset_minus", "offset_plus")
GRID = (1.58, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0)           # the bit-widths of this paper
SWEEP = tuple(range(3, 13))                            # Kumar et al. Sec. 4: P_w in [3, 12]
SWEEP_RANKING = list(SWEEP)


def neff_fraction(p: float, reading: str, gamma: float = GAMMA_W, n: float = N_W) -> float:
    """f(P) = N_eff / N under one reading of the offset."""
    if reading == "main_text":
        return 1.0 - math.exp(-p / gamma)
    if reading == "offset_minus":
        return 1.0 - math.exp(-(p / gamma - n))
    if reading == "offset_plus":
        return 1.0 - math.exp(-(p / gamma + n))
    raise ValueError(reading)


def per_bit(p: float, reading: str, gamma: float = GAMMA_W, n: float = N_W) -> float:
    """f(P) / P, proportional to N_eff at fixed weight memory (N = M / P)."""
    return neff_fraction(p, reading, gamma, n) / p


def predicted_loss(memory_bits: float, p: float, tokens: float, reading: str,
                   gamma: float = GAMMA_W, n: float = N_W) -> float:
    """Eq. 3 with N = memory / P and the constants of Table 2."""
    n_eff = memory_bits / p * neff_fraction(p, reading, gamma, n)
    if n_eff <= 0:
        raise ValueError(f"N_eff <= 0 at P = {p} under {reading}: the law is outside its domain")
    return A * n_eff ** (-ALPHA) + B * tokens ** (-ALPHA) + E


def ranking(values: dict[float, float]) -> list[float]:
    """Precisions from best (largest) to worst."""
    return [p for p, _ in sorted(values.items(), key=lambda kv: -kv[1])]


def interior_maximum(reading: str, lo: float, hi: float) -> float | None:
    """argmax of f(P)/P over a 20001-point scan of [lo, hi] if it is strictly inside, else None;
    the resolution is (hi - lo) / 20000."""
    xs = [lo + (hi - lo) * i / 20000 for i in range(20001)]
    vals = []
    for x in xs:
        f = neff_fraction(x, reading)
        vals.append(f / x if f > 0 else -math.inf)
    i = max(range(len(xs)), key=vals.__getitem__)
    return xs[i] if 0 < i < len(xs) - 1 else None


def derive() -> dict:
    out: dict = {"source": SOURCE,
                 "constants": {"A": A, "alpha": ALPHA, "B": B, "E": E, "gamma_w": GAMMA_W,
                               "n_w": N_W, "gamma_w_floating_point": GAMMA_W_FP,
                               "n_w_floating_point": N_W_FP},
                 "status": "derived by me from the published law and constants; "
                           "not reported by Kumar et al.",
                 "scope": "weight-only law, fixed weight memory N*P, fixed D, integer-type "
                          "fit, the weight precisions 3 to 12 bits of their sweep"}
    out["per_bit_by_reading"] = {
        r: {f"{p:g}": per_bit(p, r) for p in GRID} for r in READINGS}
    out["ranking_grid"] = {r: [f"{p:g}" for p in ranking({p: per_bit(p, r) for p in GRID})]
                           for r in READINGS}
    sweep_rank = {}
    for r in READINGS:
        sweep_rank[r] = ranking({float(p): per_bit(p, r) for p in SWEEP})
    sweep_rank["floating_point_gamma_no_offset"] = ranking(
        {float(p): per_bit(p, "main_text", GAMMA_W_FP) for p in SWEEP})
    sweep_rank["floating_point_gamma_offset_minus"] = ranking(
        {float(p): per_bit(p, "offset_minus", GAMMA_W_FP, N_W_FP) for p in SWEEP})
    sweep_rank["floating_point_gamma_offset_plus"] = ranking(
        {float(p): per_bit(p, "offset_plus", GAMMA_W_FP, N_W_FP) for p in SWEEP})
    out["ranking_sweep_3_to_12"] = {k: [int(p) for p in v] for k, v in sweep_rank.items()}
    # ratios of N_eff at equal memory
    out["neff_ratio"] = {
        f"{a}_over_{b}": {r: per_bit(a, r) / per_bit(b, r) for r in READINGS}
        for a, b in ((3, 6), (4, 6))}
    out["interior_maximum_of_per_bit"] = {
        r: interior_maximum(r, 1.0, 16.0) for r in READINGS}
    # the full loss, not only f/P, at fixed memory and tokens: same ranking
    loss_rank = {}
    for r in READINGS:
        for memory in (2.6e5, 8.3e5, 4.2e6, 1e8, 1e10):
            for tokens in (2e9, 1.3e10):
                try:
                    loss_rank[f"{r}/{memory:g}/{tokens:g}"] = [
                        int(p) for p in sorted(SWEEP, key=lambda p: predicted_loss(
                            memory, p, tokens, r))]
                except ValueError:
                    pass
    out["ranking_by_predicted_loss_sweep"] = loss_rank
    check(out)
    return out


def check(d: dict) -> None:
    """The statements the papers make about this derivation, as assertions."""
    for r, rank in d["ranking_sweep_3_to_12"].items():
        if rank != SWEEP_RANKING:
            raise AssertionError(f"in-range ranking under {r} is {rank}, not {SWEEP_RANKING}")
    for key, rank in d["ranking_by_predicted_loss_sweep"].items():
        if rank != SWEEP_RANKING:
            raise AssertionError(f"predicted-loss ranking {key} is {rank}")
    if len(d["ranking_by_predicted_loss_sweep"]) < 20:
        raise AssertionError("too few (memory, tokens) cases survived the domain check")
    for r in ("main_text", "offset_plus"):
        if d["interior_maximum_of_per_bit"][r] is not None:
            raise AssertionError(f"f/P has an interior maximum under {r}")
    im = d["interior_maximum_of_per_bit"]["offset_minus"]
    if im is not None and not 1.0 < im < 3.0:
        raise AssertionError(f"offset_minus maximum at {im}, expected below the fitted range")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=REPO / "results" / "kumar_derivation.json")
    args = ap.parse_args(argv)
    d = derive()
    args.out.write_text(json.dumps(d, indent=2) + "\n")
    for r in READINGS:
        print(f"{r:13s} grid ranking {' > '.join(d['ranking_grid'][r])}; "
              f"interior maximum of f/P: {d['interior_maximum_of_per_bit'][r]}")
    print("sweep 3-12 ranking under every reading:", d["ranking_sweep_3_to_12"]["main_text"])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
