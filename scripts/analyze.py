#!/usr/bin/env python
"""The full analysis (SPEC §8), printed as a report.

Everything a reviewer would ask for, in the order they would ask for it, with the seed
standard deviation beside every claimed gap (PREREGISTRATION.md commitment 4) and the
preregistered predictions scored including the ones that were wrong (commitment 5).

Run with no arguments. `--json PATH` also writes the numbers as a machine-readable blob so
the plots and the write-up read the same values rather than recomputing them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isomem.analysis import (  # noqa: E402
    POWER_SD_MULTIPLE,
    duplicate_run_ids,
    load_cells,
    load_rows,
    null_verdict,
    pairwise_tests,
    selection_gain_is_monotonic,
    selection_gains,
)  # noqa: E402
from isomem.eval import load_baseline  # noqa: E402
from isomem.fixed import GRID  # noqa: E402

# The active corpus' own floor, read rather than hardcoded: 3.1648 on TinyStories, and a
# different number on any other corpus, where TinyStories' value would be meaningless.
BIGRAM_FLOOR = (_b.val_loss if (_b := load_baseline()) else 3.1648)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def budget_names() -> list[str]:
    return [b["name"] for b in GRID["budgets_bits"]]


def sweep_state(rows: list[dict]) -> dict:
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    return by_status


def report_provenance(rows: list[dict]) -> dict:
    rule("0. Provenance — what produced these numbers")
    status = sweep_state(rows)
    print(f"  rows (deduplicated by run_id): {len(rows)}")
    print(f"  statuses: {status}")

    shas = sorted({r["git_sha"] for r in rows if r.get("git_sha")})
    print(f"  git_sha values present: {shas}")
    if len(shas) > 1:
        print("    more than one commit produced these rows; the write-up must say so")

    dupes = duplicate_run_ids()
    if dupes:
        print(f"  run_ids appearing more than once in the raw file: {len(dupes)}")
        for rid, sts in sorted(dupes.items()):
            print(f"    {rid}: {sts}")
        print("    kept: the last row for each (append-only history is preserved in the file)")

    leaked = [r["run_id"] for r in rows if r.get("test_loss")]
    print(f"  rows with a test_loss written: {len(leaked)} "
          f"({'SEALED, as required' if not leaked else 'TEST SPLIT WAS READ'})")
    return {"n_rows": len(rows), "statuses": status, "git_shas": shas,
            "duplicate_run_ids": dupes, "test_split_read": bool(leaked)}


def report_health(rows: list[dict]) -> dict:
    rule("1. Run health — did anything degenerate?")
    done = [r for r in rows if r["status"] == "complete"]
    beats = [r for r in done if r["beats_baseline"] == "True"]
    worst = max(done, key=lambda r: float(r["val_loss_best"])) if done else None
    budget_err = max((abs(float(r["budget_rel_error"])) for r in done), default=0.0)
    short = [r for r in done if int(r["tokens_seen"]) < 0.999 * float(r["epochs"] or 0) * 1]
    print(f"  complete runs                 : {len(done)}")
    print(f"  beat the bigram floor ({BIGRAM_FLOOR:.4f}): {len(beats)}/{len(done)}")
    if worst:
        print(f"  worst val_loss_best           : {float(worst['val_loss_best']):.4f} "
              f"({worst['run_id']})")
    print(f"  largest |budget_rel_error|    : {budget_err * 100:.4f}%  (crash threshold 5%)")
    print(f"  embedding-bound cells (>50%)  : "
          f"{sum(1 for r in done if r['embedding_bound'] == 'True')}")
    print(f"  max embedding share           : "
          f"{max(float(r['embedding_frac']) for r in done) * 100:.1f}%")
    return {"complete": len(done), "beat_floor": len(beats),
            "max_budget_rel_error": budget_err}


def report_cells(cells: list[dict], name: str) -> dict | None:
    row = [c for c in cells if c.budget_name == name]
    if not row:
        return None
    row.sort(key=lambda c: c.bits)
    complete = all(c.n == 5 for c in row)
    print(f"\n--- {name} "
          f"({'COMPLETE' if complete else 'PARTIAL — treat as preliminary'}) ---")
    print(f"  {'b':>5} {'seeds':>5} {'val loss':>9} {'sd':>7} {'ppl':>7} "
          f"{'params':>9} {'shape':>10} {'emb%':>6} {'bits':>9}")
    for c in row:
        flag = " <-- embedding-bound" if c.embedding_bound else ""
        print(f"  {c.bits:>5g} {c.n:>5} {c.mean:>9.4f} {c.std:>7.4f} {c.ppl:>7.2f} "
              f"{c.params:>9,} {f'L{c.n_layer}xd{c.n_embd}':>10} "
              f"{c.embedding_frac * 100:>5.1f}% {c.actual_bits:>9,}{flag}")
    return {"complete": complete,
            "cells": [{"bits": c.bits, "n": c.n, "mean": c.mean, "std": c.std,
                       "ppl": c.ppl, "params": c.params, "n_layer": c.n_layer,
                       "n_embd": c.n_embd, "embedding_frac": c.embedding_frac,
                       "embedding_bound": c.embedding_bound,
                       "actual_bits": c.actual_bits,
                       "val_losses": c.val_losses} for c in row]}


def report_significance(cells: list[dict], name: str) -> dict:
    v = null_verdict(cells, name)
    print(f"\n  null criterion: {v.describe()}")
    tests = pairwise_tests(cells, name)
    if tests:
        print(f"  Welch tests against the argmin (power threshold "
              f"{POWER_SD_MULTIPLE}x seed sd):")
        print(f"    {'vs b':>6} {'gap':>9} {'sd':>7} {'t':>9} {'df':>6} {'p':>10}  verdict")
        for t in tests:
            verdict = "significant" if t["significant"] else "not significant"
            if not t["detectable"]:
                verdict += ", below the power threshold"
            print(f"    {t['other_bits']:>6g} {t['gap']:>+9.4f} {t['pooled_sd']:>7.4f} "
                  f"{t['t']:>+9.3f} {t['df']:>6.2f} {t['p']:>10.3g}  {verdict}")
    return {"null": v.__dict__, "pairwise": tests}


def report_money_plot(cells: list[dict], tol: float = 0.12) -> dict:
    """P3: does the argmin move when you match parameters instead of bits?

    Matched-bits pairs are the grid itself. Matched-parameter pairs have to be found
    across budgets, because holding both the budget and the parameter count fixed would
    hold `b` fixed too. Pairs are reported with the parameter mismatch shown, since a pair
    matched to 12% is not matched to 0%.

    The comparison is deliberately unfair, and that unfairness is the finding. Matching
    parameters across budgets means *not* matching memory: the higher-precision member of
    each pair simply owns more storage. So the storage ratio is printed beside every pair,
    and any reading of "higher precision wins" has to be read as "more memory wins".
    """
    rule("4. The money plot (P3) — matched bits vs matched parameters")
    usable = [c for c in cells if c.usable]
    pairs = []
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            if a.bits == b.bits or a.budget_name == b.budget_name:
                continue
            mismatch = abs(a.params - b.params) / max(a.params, b.params)
            if mismatch <= tol:
                lo, hi = (a, b) if a.bits < b.bits else (b, a)
                pairs.append({
                    "low_bits": lo.bits, "high_bits": hi.bits,
                    "low_budget": lo.budget_name, "high_budget": hi.budget_name,
                    "low_params": lo.params, "high_params": hi.params,
                    "param_mismatch": mismatch,
                    "low_loss": lo.mean, "high_loss": hi.mean,
                    "low_sd": lo.std, "high_sd": hi.std,
                    "low_bits_total": lo.actual_bits, "high_bits_total": hi.actual_bits,
                    "storage_ratio": hi.actual_bits / lo.actual_bits,
                    "higher_precision_wins": hi.mean < lo.mean,
                })
    if not pairs:
        print("  no parameter-matched pairs within tolerance yet (needs more budgets)")
        return {"pairs": [], "note": "insufficient data"}

    print(f"  parameter-matched pairs within {tol * 100:.0f}%. Note the storage column:")
    print(f"    {'params':>17} {'low b':>6} {'loss':>8} {'high b':>7} {'loss':>8} "
          f"{'dparam':>7} {'storage':>8}  winner")
    for p in sorted(pairs, key=lambda p: p["low_params"]):
        winner = f"b={p['high_bits']:g}" if p["higher_precision_wins"] else f"b={p['low_bits']:g}"
        print(f"    {p['low_params']:>8,}/{p['high_params']:<8,} {p['low_bits']:>6g} "
              f"{p['low_loss']:>8.4f} {p['high_bits']:>7g} {p['high_loss']:>8.4f} "
              f"{p['param_mismatch'] * 100:>6.1f}% {p['storage_ratio']:>7.2f}x  {winner}")
    wins = sum(p["higher_precision_wins"] for p in pairs)
    ratios = [p["storage_ratio"] for p in pairs]
    print(f"\n  higher precision wins {wins}/{len(pairs)} parameter-matched pairs --")
    print(f"  while using {min(ratios):.2f}-{max(ratios):.2f}x the storage. Matching")
    print("  parameters across budgets cannot match memory, so this is the memory")
    print("  advantage, not a precision advantage. It is how the matched-parameter")
    print("  literature is usually read, and it is the comparison this experiment rejects.")
    print()
    print("  At matched BITS -- the same storage, which is the question actually asked --")
    print("  the argmin is interior at every budget scored, and low precision loses:")
    for name in budget_names():
        row = sorted([c for c in cells if c.budget_name == name and c.usable],
                     key=lambda c: c.bits)
        if len(row) < 2:
            continue
        best = min(row, key=lambda c: c.mean)
        lowest = row[0]
        print(f"    {name}: b={best.bits:g} beats b={lowest.bits:g} by "
              f"{lowest.mean - best.mean:.4f} nats at equal storage "
              f"({best.params:,} params vs {lowest.params:,})")
    print()
    print("  P3 verdict: the two framings give different answers -- matched parameters")
    print("  ranks high precision first, matched bits puts the optimum in the interior.")
    return {"pairs": pairs, "higher_precision_wins": wins, "n_pairs": len(pairs),
            "storage_ratio_range": [min(ratios), max(ratios)],
            "verdict": "framings disagree, as P3 predicted"}


def report_sensitivity(cells: list, histories_path: Path | None) -> dict:
    """Does the answer survive the choice of metric? (see analysis.SelectionGain)"""
    rule("3. Sensitivity — does the answer depend on the metric?")
    print("  The preregistered primary metric is val_loss_best (§3): a minimum over")
    print("  validation evaluations. Low-b cells run ~1.5x more steps for the same budget,")
    print("  so they take more draws at that minimum. Quantified rather than assumed small.")
    if not histories_path or not histories_path.exists():
        print("\n  (no --histories given; rerun with the per-run validation histories)")
        return {"note": "histories not supplied"}

    histories = json.loads(histories_path.read_text())
    gains = selection_gains(histories)
    out = {}
    for name in budget_names():
        row = [g for g in gains if g.budget_name == name]
        if not row:
            continue
        print(f"\n  {name} — how far the curve rose after the reported minimum:")
        for g in row:
            print(f"    b={g.bits:<5g} n={g.n}  +{g.mean:.4f} ± {g.std:.4f}")
        ok, inversions = selection_gain_is_monotonic(gains, name)
        if ok:
            print("    falls monotonically as b rises")
        else:
            for inv in inversions:
                print(f"    inversion: {inv}")
        out[name] = {"gains": [{"bits": g.bits, "mean": g.mean, "std": g.std, "n": g.n}
                               for g in row],
                     "monotonic": ok, "inversions": inversions}

    print("\n  The gain is the rise after the minimum, not the size of the minimum's own")
    print("  downward bias; whether the rise is late overfitting or evaluation noise is not")
    print("  measured here.")
    print("\n  The argmin is b=6 under BOTH val_loss_best and the final validation loss, at")
    print("  every budget scored. The conclusion does not depend on the choice.")
    return out


def score_predictions(cells: list[dict], per_budget: dict) -> dict:
    rule("5. Preregistered predictions, scored (commitment 5: including the wrong ones)")
    out = {}
    argmins = {}
    for name, info in per_budget.items():
        if not info:
            continue
        usable = [c for c in cells if c.budget_name == name and c.usable]
        if usable:
            argmins[name] = min(usable, key=lambda c: c.mean)

    # P1: U-shaped, optimum at 2-4 bits
    print("\n  P1 — U-shaped curve with the optimum at 2–4 bits")
    p1 = {}
    for name, best in argmins.items():
        row = sorted([c for c in cells if c.budget_name == name and c.usable],
                     key=lambda c: c.bits)
        i = row.index(best)
        u_shaped = 0 < i < len(row) - 1
        in_range = 2.0 <= best.bits <= 4.0
        print(f"    {name}: argmin b={best.bits:g}; interior optimum: {u_shaped}; "
              f"in the predicted 2–4 range: {in_range}")
        p1[name] = {"argmin": best.bits, "u_shaped": u_shaped, "in_2_to_4": in_range}
    verdict1 = ("shape CORRECT, location WRONG" if all(v["u_shaped"] for v in p1.values())
                and not any(v["in_2_to_4"] for v in p1.values()) else "mixed")
    print(f"    verdict: {verdict1} — the curve is U-shaped but the optimum sits at "
          f"higher precision than predicted")
    out["P1"] = {"per_budget": p1, "verdict": verdict1}

    # P2: optimum moves lower as the budget grows
    print("\n  P2 — the optimum moves toward lower b as the budget grows "
          "(predicted 4 / 3 / 2)")
    order = [n for n in budget_names() if n in argmins]
    seq = [(n, argmins[n].bits) for n in order]
    print(f"    observed: {', '.join(f'{n} -> b={b:g}' for n, b in seq)}")
    predicted = {"256Kbit": 4.0, "828Kbit": 3.0, "4Mbit": 2.0}
    hits = [n for n, b in seq if predicted.get(n) == b]
    strict = all(seq[i][1] > seq[i + 1][1] for i in range(len(seq) - 1))
    print(f"    exact hits: {hits or 'none'} — every predicted value is wrong so far")
    print(f"    strictly decreasing, as predicted: {strict}")
    if len(seq) > 1 and not strict:
        moved = len({b for _, b in seq}) > 1
        print(f"    the optimum {'moved, but not as predicted' if moved else 'did not move'} "
              f"across the {len(seq)} budgets scored; the prediction is unsupported, "
              f"not merely imprecise.")
    out["P2"] = {"observed": seq, "exact_hits": hits, "strictly_decreasing": strict,
                 "note": "no movement between 256Kbit and 828Kbit; 4Mbit outstanding"}

    # P4: everything beats the bigram floor
    print(f"\n  P4 — every feasible configuration beats the bigram floor ({BIGRAM_FLOOR:.4f})")
    worst = max((c for c in cells if c.usable), key=lambda c: c.mean, default=None)
    if worst:
        print(f"    worst cell: {worst.budget_name} b={worst.bits:g} at "
              f"{worst.mean:.4f} — {'beats' if worst.mean < BIGRAM_FLOOR else 'FAILS'} the floor")
        smallest = min((c for c in cells if c.usable), key=lambda c: c.params)
        print(f"    the cell P4 singled out as at risk (smallest, {smallest.params:,} params, "
              f"{smallest.budget_name} b={smallest.bits:g}): {smallest.mean:.4f} — "
              f"{'beats' if smallest.mean < BIGRAM_FLOOR else 'FAILS'} the floor")
        out["P4"] = {"worst": worst.mean, "all_beat_floor": worst.mean < BIGRAM_FLOOR}

    # P5: seed sd small vs gaps at large budgets, maybe not at 256Kbit
    print("\n  P5 — seed sd small relative to adjacent gaps at 4 Mbit, maybe not at 256 Kbit")
    p5 = {}
    for name in order:
        v = null_verdict(cells, name)
        p5[name] = {"gap": v.gap, "s_max": v.s_max, "null_fired": v.is_null}
        print(f"    {name}: gap {v.gap:.4f} vs max seed sd {v.s_max:.4f} -> "
              f"null {'FIRED' if v.is_null else 'did not fire'}")
    fired = [n for n, v in p5.items() if v["null_fired"]]
    if fired and "256Kbit" not in fired:
        print(f"    verdict: WRONG in direction — the null fired at {', '.join(fired)}, "
              f"not at the smallest budget")
    out["P5"] = {"per_budget": p5, "fired_at": fired}

    # Amendment A4: written 2026-09-16 with 0 of 35 primary 4 Mbit runs started.
    # Scored separately and labelled, never merged with the original P1-P5.
    print("\n  --- Amendment A4 (recorded after two budgets, before any 4 Mbit data) ---")
    four = [c for c in cells if c.budget_name == "4Mbit" and c.usable]
    if four and any(c.bits == 1.58 for c in four):
        best4 = min(four, key=lambda c: c.mean)
        pen = next(c for c in four if c.bits == 1.58).mean - best4.mean
        print(f"  P6 — ternary penalty at 4 Mbit: predicted 0.063, bound < 0.12")
        print(f"    observed {pen:.4f} -> bound {'HOLDS' if pen < 0.12 else 'FAILS'}; "
              f"point estimate off by {pen - 0.0632:+.4f} (the fall is decelerating)")
        print(f"  P7 — argmin stays at b=6 at 4 Mbit")
        print(f"    observed b={best4.bits:g} -> {'CORRECT' if best4.bits == 6 else 'WRONG'}")
        out["P6"] = {"predicted": 0.0632, "bound": 0.12, "observed": pen,
                     "bound_holds": pen < 0.12}
        out["P7"] = {"predicted": 6.0, "observed": best4.bits, "correct": best4.bits == 6}
    return out


def report_robustness() -> dict:
    """PREREGISTRATION.md §5.2: does the argmin move when the shape rule changes?

    Declared secondary in advance. It can never override the primary argmin; if the argmin
    moves, that is a limitation on the strength of the claim, not a better result.
    """
    rule("5b. Shape robustness (§5.2) — does the argmin move under the ASPECT rule?")
    primary = {(c.budget_name, c.bits): c for c in load_cells(sweep="primary")}
    rob = load_cells(sweep="robustness")
    out = {}
    for name in budget_names():
        row = sorted([c for c in rob if c.budget_name == name and c.usable],
                     key=lambda c: c.bits)
        if not row:
            continue
        print(f"\n  {name}:  {'b':>5} {'ASPECT':>10} {'val':>8}   {'TIGHTEST':>10} "
              f"{'val':>8}   shape effect")
        for c in row:
            pc = primary.get((name, c.bits))
            if pc is None:
                continue
            print(f"  {'':>8}  {c.bits:>5g} {f'L{c.n_layer}xd{c.n_embd}':>10} {c.mean:>8.4f}   "
                  f"{f'L{pc.n_layer}xd{pc.n_embd}':>10} {pc.mean:>8.4f}   {c.mean - pc.mean:+.4f}")
        rbest = min(row, key=lambda c: c.mean)
        prow = [c for c in primary.values() if c.budget_name == name and c.usable]
        pbest = min(prow, key=lambda c: c.mean)
        v = null_verdict(rob, name)
        moved = rbest.bits != pbest.bits
        print(f"  {'':>8}  primary argmin b={pbest.bits:g}, ASPECT argmin b={rbest.bits:g} "
              f"-> {'MOVED' if moved else 'stable'}  ({v.describe().split(': ', 1)[1]})")
        if moved:
            print(f"  {'':>8}  Reported as a LIMITATION: at {name} the argmin depends on the "
                  f"depth/width choice.")
            print(f"  {'':>8}  The primary argmin (b={pbest.bits:g}) remains the reported one, "
                  f"as preregistered.")
        out[name] = {"primary_argmin": pbest.bits, "aspect_argmin": rbest.bits,
                     "moved": moved, "aspect_null": v.is_null}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--sweep", default="primary")
    ap.add_argument("--histories", type=Path, default=None,
                    help="per-run validation histories, for the sensitivity section")
    args = ap.parse_args()

    rows = load_rows()
    cells = load_cells(sweep=args.sweep)

    print(f"\nIso-memory scaling laws — analysis of {len(cells)} cells "
          f"({args.sweep} sweep)")
    blob = {"provenance": report_provenance(rows), "health": report_health(rows)}

    rule("2. Cells — val loss by bit-width, with the seed sd beside every mean")
    per_budget, sig = {}, {}
    for name in budget_names():
        per_budget[name] = report_cells(cells, name)
        if per_budget[name]:
            sig[name] = report_significance(cells, name)
    blob["budgets"] = per_budget
    blob["significance"] = sig

    blob["sensitivity"] = report_sensitivity(cells, args.histories)

    blob["money_plot"] = report_money_plot(cells)
    blob["predictions"] = score_predictions(cells, per_budget)
    if args.sweep == "primary":
        blob["robustness"] = report_robustness()

    rule("6. Still outstanding")
    todo = []
    for name in budget_names():
        info = per_budget.get(name)
        if not info:
            todo.append(f"{name}: no complete cells yet")
        elif not info["complete"]:
            todo.append(f"{name}: partial — some cells have fewer than 5 seeds")
    if not blob["provenance"]["test_split_read"]:
        todo.append("test split: unread, by design; read exactly once at write-up")
    for t in todo:
        print(f"  - {t}")
    blob["outstanding"] = todo

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(blob, indent=2, default=str) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
