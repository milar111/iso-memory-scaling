#!/usr/bin/env python
"""Do runs that the experiment makes identical reproduce each other exactly?

Reads ``results/results.csv`` and ``results/histories.json`` (and ``results/sweep.log`` for the
resume markers). It trains nothing, needs no GPU, and takes no test-split value from
the CSV: only the columns in ``USED_COLUMNS`` are kept from each row, and the test columns are
not among them (asserted). The CSV reader parses whole rows; the others are dropped unused.

What is compared
----------------
A *twin pair* is two complete runs, one from the primary grid (shape rule TIGHTEST) and one
from the shape-robustness grid (shape rule ASPECT, run id ``<primary id>-aspect``), whose
resolved architecture and seed are identical. The criterion, asserted in ``find_pairs``, is
equality of all of ``ARCH_KEY``: budget, bits per weight, layers, width, heads, MLP ratio,
vocabulary, counted parameters, norm parameters, and seed. Within a pair, the training set-up
(``SETUP_KEYS``: steps, tokens seen, learning rate, micro-batch, accumulation, effective batch,
fixed-config hash) must also be identical, or the script raises. Because the two shape rules
only coincide for some cells, most robustness runs have a different architecture and so no
twin; they are counted, not compared.

For each pair the complete evaluation histories are compared: every evaluation step, and the
train and validation loss at each, with exact float equality, and the maximum absolute
difference. ``val_loss_best``, ``train_loss`` and ``best_step`` of the CSV are compared too.

Resumed runs are taken from ``results/sweep.log``: a run counts as resumed if the log has a
``(resuming)`` header for it, and as stopped if the log has a ``stopped at step`` line for it.
The results table itself carries no resume marker. A pair in which exactly one member was
resumed compares a run that crossed a checkpoint with one that did not.

What this does NOT show
-----------------------
* Runs without a twin: nothing here says anything about them. Determinism is verified for the
  cells that have a twin pair, not for all cells.
* The pairs ran on one device kind (the CSV ``device`` column, reported in the output) with one
  framework build and different commits of the code (``git_sha`` of each member is reported); a
  bit-identical pair shows that the training code of those commits is deterministic for these
  cells, not that another GPU or driver would give the same numbers.
* "Uninterrupted" means that the sweep log has no stop or resume marker for the run; an
  interruption the log does not record would not be seen.
* Only resumed runs that have a twin are compared. Other runs were resumed too (listed under
  ``resumed_without_twin``); their resumes are not checked by this script.
* The 20-step same-seed repeat in ``results/bench_*.json`` is separate evidence (one final
  train loss on random tokens); the CPU resume unit test ``test_resume_bitwise`` checks
  agreement only to 1e-5. Neither is part of this comparison.
* The 256 Kbit, 8-bit rerun recorded in a commit message is not stored and not covered.

Writes ``results/determinism_check.json``.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ARCH_KEY = ("budget_name", "bits_per_weight", "n_layer", "n_embd", "n_head", "mlp_ratio",
            "vocab", "params_counted", "norm_params", "seed")
SETUP_KEYS = ("steps", "tokens_seen", "lr", "micro_batch", "grad_accum",
              "effective_batch_tokens", "fixed_config_hash")
ID_COLUMNS = ("run_id", "status", "sweep", "shape_rule", "device", "torch_version",
              "cuda_version", "git_sha")
# Final-step quantities of the training and validation streams only; no test column.
RESULT_COLUMNS = ("train_loss", "val_loss_best", "best_step")
USED_COLUMNS = tuple(dict.fromkeys(ID_COLUMNS + ARCH_KEY + SETUP_KEYS + RESULT_COLUMNS))
assert not any("test" in c for c in USED_COLUMNS), "the test split must not be read"

_RESUME_RE = re.compile(r"^\[\d+/\d+\]\s+(\S+)\s.*\(resuming\)")
_STOP_RE = re.compile(r"^\s+(\S+): stopped at step (\d+)/(\d+) \((\w+)\)")


def read_results(path: Path) -> list[dict]:
    """Rows of the results table, restricted to ``USED_COLUMNS``."""
    with path.open(newline="") as f:
        return [{c: r[c] for c in USED_COLUMNS} for r in csv.DictReader(f)]


def read_resume_markers(path: Path) -> dict[str, dict]:
    """Per run id: number of ``(resuming)`` headers and of ``stopped at step`` lines."""
    out: dict[str, dict] = collections.defaultdict(
        lambda: {"n_resume_headers": 0, "n_stops": 0, "stops": []})
    for line in path.read_text().splitlines():
        m = _RESUME_RE.match(line)
        if m:
            out[m.group(1)]["n_resume_headers"] += 1
            continue
        m = _STOP_RE.match(line)
        if m:
            e = out[m.group(1)]
            e["n_stops"] += 1
            e["stops"].append({"step": int(m.group(2)), "of": int(m.group(3)),
                               "reason": m.group(4)})
    return dict(out)


def find_pairs(rows: list[dict]) -> tuple[list[tuple[dict, dict]], dict]:
    """Twin pairs among the complete rows, by the criterion in the module docstring.

    Raises when a group of equal ``ARCH_KEY`` is not exactly one primary run and its ``-aspect``
    robustness run, or when the two differ in ``SETUP_KEYS``.
    """
    complete = [r for r in rows if r["status"] == "complete"]
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in complete:
        groups[tuple(r[k] for k in ARCH_KEY)].append(r)
    pairs = []
    for key, g in groups.items():
        if len(g) == 1:
            continue
        assert len(g) == 2, f"{len(g)} complete runs share the architecture {key}: " \
                            f"{[r['run_id'] for r in g]}"
        prim = [r for r in g if r["sweep"] == "primary" and r["shape_rule"] == "TIGHTEST"]
        rob = [r for r in g if r["sweep"] == "robustness" and r["shape_rule"] == "ASPECT"]
        assert len(prim) == 1 and len(rob) == 1, \
            f"a twin group is not one primary and one robustness run: {[r['run_id'] for r in g]}"
        assert rob[0]["run_id"] == prim[0]["run_id"] + "-aspect", \
            f"unexpected twin ids {prim[0]['run_id']} / {rob[0]['run_id']}"
        for k in SETUP_KEYS:
            assert prim[0][k] == rob[0][k], \
                f"{prim[0]['run_id']} and {rob[0]['run_id']} differ in {k}: " \
                f"{prim[0][k]} vs {rob[0][k]}"
        pairs.append((prim[0], rob[0]))
    stats = {
        "n_complete_runs": len(complete),
        "n_complete_primary": sum(r["sweep"] == "primary" for r in complete),
        "n_complete_robustness": sum(r["sweep"] == "robustness" for r in complete),
        "n_runs_in_pairs": 2 * len(pairs),
        "n_complete_without_twin": len(complete) - 2 * len(pairs),
    }
    return pairs, stats


def _finite(x: float, what: str) -> float:
    assert isinstance(x, (int, float)) and math.isfinite(x), f"non-finite {what}: {x!r}"
    return float(x)


def compare_histories(a: dict, b: dict) -> dict:
    """Exact comparison of two history records (``{"history": [{step, train, val}], "step": n}``).

    Exact equality means: same number of evaluations, the same step at each, and the same train
    and validation loss (float ``==``) at each; ``max_abs_diff`` is the largest absolute
    difference over the common prefix of evaluations (train and validation together). Unequal
    lengths always give ``exact_equal = False``.
    """
    ha, hb = a["history"], b["history"]
    n = min(len(ha), len(hb))
    steps_equal = [e["step"] for e in ha[:n]] == [e["step"] for e in hb[:n]]
    d_train = d_val = 0.0
    n_unequal = 0
    for ea, eb in zip(ha[:n], hb[:n]):
        for field in ("train", "val"):
            x, y = _finite(ea[field], field), _finite(eb[field], field)
            if x != y:
                n_unequal += 1
            d = abs(x - y)
            if field == "train":
                d_train = max(d_train, d)
            else:
                d_val = max(d_val, d)
    same_len = len(ha) == len(hb)
    return {
        "n_evals_a": len(ha), "n_evals_b": len(hb),
        "same_length": same_len,
        "same_steps": steps_equal and same_len,
        "n_unequal_values": n_unequal,
        "final_step_equal": a["step"] == b["step"],
        "max_abs_diff_train": d_train, "max_abs_diff_val": d_val,
        "max_abs_diff": max(d_train, d_val),
        "exact_equal": (same_len and steps_equal and n_unequal == 0
                        and a["step"] == b["step"]),
    }


def compare_pair(prim: dict, rob: dict, hist: dict, markers: dict) -> dict:
    cmp_ = compare_histories(hist[prim["run_id"]], hist[rob["run_id"]])
    csv_equal = {k: float(prim[k]) == float(rob[k]) for k in RESULT_COLUMNS}
    mk = {r["run_id"]: markers.get(r["run_id"], {"n_resume_headers": 0, "n_stops": 0,
                                                 "stops": []})
          for r in (prim, rob)}
    resumed = {rid: m["n_resume_headers"] > 0 for rid, m in mk.items()}
    interrupted = {rid: (m["n_resume_headers"] + m["n_stops"]) > 0 for rid, m in mk.items()}
    return {
        "primary": prim["run_id"], "twin": rob["run_id"],
        "architecture": {k: prim[k] for k in ARCH_KEY},
        "git_sha": {"primary": prim["git_sha"], "twin": rob["git_sha"]},
        "device": {"primary": prim["device"], "twin": rob["device"]},
        "n_evals": cmp_["n_evals_a"] if cmp_["same_length"] else None,
        **cmp_,
        "csv_result_columns_equal": csv_equal,
        "sweep_log": {rid: {"resumed": resumed[rid], "interrupted": interrupted[rid],
                            "n_resume_headers": mk[rid]["n_resume_headers"],
                            "stops": mk[rid]["stops"]} for rid in mk},
        "identical": bool(cmp_["exact_equal"] and all(csv_equal.values())),
    }


def check(results_csv: Path, histories_json: Path, sweep_log: Path) -> dict:
    from isomem.fixed import git_sha

    rows = read_results(results_csv)
    hist = json.loads(histories_json.read_text())
    markers = read_resume_markers(sweep_log)
    pairs, stats = find_pairs(rows)
    for prim, rob in pairs:
        for r in (prim, rob):
            assert r["run_id"] in hist, f"no history for {r['run_id']}"
    out_pairs = [compare_pair(p, r, hist, markers) for p, r in pairs]

    in_pair = {x for p in pairs for x in (p[0]["run_id"], p[1]["run_id"])}
    complete_ids = {r["run_id"] for r in rows if r["status"] == "complete"}
    resumed_all = sorted(rid for rid, m in markers.items()
                         if m["n_resume_headers"] > 0 and rid in complete_ids)
    resumed_without_twin = [rid for rid in resumed_all if rid not in in_pair]
    resumed_pairs = [p for p in out_pairs
                     if any(v["resumed"] for v in p["sweep_log"].values())]
    one_resumed_vs_clean = [p for p in resumed_pairs
                            if sum(v["resumed"] for v in p["sweep_log"].values()) == 1
                            and sum(v["interrupted"] for v in p["sweep_log"].values()) == 1]
    ev = [p["n_evals"] for p in out_pairs if p["n_evals"] is not None]
    summary = {
        "n_pairs": len(out_pairs),
        "n_identical": sum(p["identical"] for p in out_pairs),
        "all_identical": all(p["identical"] for p in out_pairs),
        "n_cells": len({tuple(p["architecture"][k] for k in ARCH_KEY[:-1]) for p in out_pairs}),
        "seeds": sorted({int(p["architecture"]["seed"]) for p in out_pairs}),
        "n_evals_min": min(ev) if ev else None,
        "n_evals_max": max(ev) if ev else None,
        "n_evals_per_cell": sorted({
            (p["architecture"]["budget_name"], p["architecture"]["bits_per_weight"],
             p["n_evals"]) for p in out_pairs}),
        "max_abs_diff_over_pairs": max((p["max_abs_diff"] for p in out_pairs), default=None),
        "devices": sorted({r["device"] for r in rows if r["status"] == "complete"}),
        "torch_versions": sorted({r["torch_version"] for r in rows
                                  if r["status"] == "complete"}),
        "cuda_versions": sorted({r["cuda_version"] for r in rows
                                 if r["status"] == "complete"}),
        "n_pairs_with_a_resumed_member": len(resumed_pairs),
        "resumed_vs_uninterrupted_pairs": [
            {"resumed": next(rid for rid, v in p["sweep_log"].items() if v["resumed"]),
             "uninterrupted": next(rid for rid, v in p["sweep_log"].items()
                                   if not v["interrupted"]),
             "stops": next(v["stops"] for v in p["sweep_log"].values() if v["resumed"]),
             "n_evals": p["n_evals"], "identical": p["identical"],
             "max_abs_diff": p["max_abs_diff"]}
            for p in one_resumed_vs_clean],
        "n_resumed_vs_uninterrupted": len(one_resumed_vs_clean),
        "n_resumed_vs_uninterrupted_identical": sum(p["identical"] for p in one_resumed_vs_clean),
        "n_complete_runs_resumed_per_sweep_log": len(resumed_all),
        "resumed_without_twin": resumed_without_twin,
        **stats,
        "n_robustness_runs_without_identical_architecture_twin": (
            stats["n_complete_robustness"] - len(out_pairs)),
    }
    return {
        "git_sha": git_sha(),
        "inputs": {"results_csv": results_csv.name, "histories_json": histories_json.name,
                   "sweep_log": sweep_log.name},
        "criterion": {
            "pair": "two complete runs, one primary (shape rule TIGHTEST) and its robustness "
                    "(shape rule ASPECT) run '<primary>-aspect', equal in every ARCH_KEY column",
            "arch_key": list(ARCH_KEY), "setup_keys_asserted_equal": list(SETUP_KEYS),
            "equality": "float == on every evaluation's train and val loss and step; equal "
                        "number of evaluations; equal final step; equal CSV train_loss, "
                        "val_loss_best, best_step",
            "resumed_run": "a run id with a '(resuming)' header in results/sweep.log; "
                           "the results table has no resume marker",
            "columns_read_from_csv": list(USED_COLUMNS),
        },
        "summary": summary,
        "pairs": out_pairs,
        "does_not_show": [
            "Anything about the complete runs without a twin (n_complete_without_twin of them): "
            "determinism is verified for the cells that have a twin pair, not for all cells.",
            "Anything about another device, driver or framework build: all runs share the "
            "devices/torch/cuda versions listed in the summary, and the pair members come from "
            "different commits of the code.",
            "Interruptions that the sweep log does not record: 'uninterrupted' means no marker.",
            "Resumes of runs without an uninterrupted twin (resumed_without_twin) are not "
            "compared.",
            "The 20-step same-seed repeat in results/bench_*.json (separate evidence), the unit "
            "test test_resume_bitwise (agreement to 1e-5 on CPU), and the 256 Kbit 8-bit rerun "
            "recorded only in a commit message.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=REPO / "results" / "results.csv")
    ap.add_argument("--histories", type=Path, default=REPO / "results" / "histories.json")
    ap.add_argument("--log", type=Path, default=REPO / "results" / "sweep.log")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "determinism_check.json")
    args = ap.parse_args(argv)
    d = check(args.results, args.histories, args.log)
    args.out.write_text(json.dumps(d, indent=2) + "\n")
    s = d["summary"]
    for p in d["pairs"]:
        print(f"{p['primary']:26s} vs {p['twin']:32s} evals {p['n_evals']} "
              f"exact_equal {p['exact_equal']} max|diff| {p['max_abs_diff']}")
    print(f"pairs {s['n_pairs']}, identical {s['n_identical']}, evals {s['n_evals_min']}"
          f"..{s['n_evals_max']}, resumed-vs-uninterrupted {s['n_resumed_vs_uninterrupted']}"
          f" (identical {s['n_resumed_vs_uninterrupted_identical']}), devices {s['devices']}")
    print(f"wrote {args.out}")
    return 0 if s["all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
