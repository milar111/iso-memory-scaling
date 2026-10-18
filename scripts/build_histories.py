#!/usr/bin/env python
"""Build ``results/histories.json`` from the run checkpoints.

Every complete run's checkpoint (``runs/<run_id>.pt``, written by ``isomem.train.train_one``)
holds its full validation history and its final step. This script collects them into one
JSON file, ``{run_id: {"history": [...], "step": int}}``, which the analysis, the figures and
``scripts/paper_numbers.py`` read.

``--check`` builds from whatever checkpoints are present and compares each with the entry
already in ``results/histories.json`` instead of writing, so a copy of the file can be
verified against the checkpoints it came from without holding all of them.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
RESULTS_CSV = REPO / "results" / "results.csv"
OUT = REPO / "results" / "histories.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, default=REPO / "runs")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true",
                    help="compare with the existing --out file instead of writing it")
    args = ap.parse_args(argv)

    rows = {}
    for r in csv.DictReader(open(RESULTS_CSV)):
        rows[r["run_id"]] = r  # the last row per run_id, as the analysis keeps
    built = {}
    for ckpt in sorted(args.runs.glob("*.pt")):
        r = rows.get(ckpt.stem)
        if r is None or r["status"] != "complete":
            continue
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if int(state["step"]) != int(r["steps"]):
            raise SystemExit(f"{ckpt.stem}: checkpoint at step {state['step']}, but the run "
                             f"completed at {r['steps']}")
        built[ckpt.stem] = {"history": state["history"], "step": int(state["step"])}

    if args.check:
        have = json.loads(args.out.read_text())
        bad = [k for k, v in built.items() if have.get(k) != v]
        print(f"checked {len(built)} checkpoints against {args.out}: "
              f"{len(built) - len(bad)} identical, {len(bad)} different {bad}")
        return 1 if bad else 0

    complete = {k for k, r in rows.items() if r["status"] == "complete"}
    missing = sorted(complete - set(built))
    if missing:
        raise SystemExit(f"{len(missing)} complete runs have no final checkpoint under "
                         f"{args.runs}, e.g. {missing[:3]}; refusing to write a partial file")
    args.out.write_text(json.dumps(built) + "\n")
    print(f"wrote {args.out} ({len(built)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
