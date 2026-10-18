#!/usr/bin/env python
"""Does storing the RMSNorm gains in bf16 change the validation loss? (Deviation D6)

The accounting counts every RMSNorm gain at 16 bits, but training and every reported
evaluation hold and apply the gains in fp32. This script measures what rounding the gains
to bf16 does to the loss of a finished model, so the 16-bit count is backed by a number
rather than an argument.

For every checkpoint under ``runs/`` that belongs to a complete run of ``results.csv`` and
holds that run's final step, it evaluates the validation loss twice on the same device,
through ``isomem.train.evaluate`` (the fixed 50 validation batches every run is scored on):

* ``fp32``: the checkpoint as trained (gains in fp32);
* ``bf16``: the same weights with every RMSNorm gain rounded to bf16 (and held in an fp32
  tensor, so nothing else changes).

It also records the final validation loss the run itself logged (``history[-1]``), so the
reproduction error of this device against the training device is visible next to the
effect being measured. A checkpoint holds the FINAL-step weights, not the weights at
``val_loss_best``, so ``val_loss_best`` itself cannot be re-evaluated this way.

Reads only the validation split (``isomem.data.load_tokens("val")``). Writes
``results/gain_bf16_check.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from isomem import data as D  # noqa: E402
from isomem.accounting import ModelConfig  # noqa: E402
from isomem.fixed import FIXED  # noqa: E402
from isomem.model import IsoMemGPT, RMSNorm  # noqa: E402
from isomem.train import evaluate, make_batch_schedule, resolve_device  # noqa: E402

RESULTS_CSV = REPO / "results" / "results.csv"
OUT = REPO / "results" / "gain_bf16_check.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, default=REPO / "runs")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    rows = {}
    for r in csv.DictReader(open(RESULTS_CSV)):
        rows[r["run_id"]] = r  # last row per run_id, as the analysis keeps
    device = resolve_device()
    t_cfg = FIXED["training"]
    val = D.load_tokens("val")

    out = []
    for ckpt in sorted(args.runs.glob("*.pt")):
        run_id = ckpt.stem
        r = rows.get(run_id)
        if r is None or r["status"] != "complete":
            print(f"skip {run_id}: not a complete run in results.csv")
            continue
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if int(state["step"]) != int(r["steps"]):
            print(f"skip {run_id}: checkpoint at step {state['step']}, run ended at {r['steps']}")
            continue
        cfg = ModelConfig(n_layer=int(r["n_layer"]), n_embd=int(r["n_embd"]),
                          mlp_ratio=float(r["mlp_ratio"]), vocab_size=int(r["vocab"]),
                          block_size=FIXED["model"]["block_size"])
        bits = float(r["bits_per_weight"])
        model = IsoMemGPT(cfg, bits, rope_theta=FIXED["model"]["rope_theta"]).to(device)
        model.load_state_dict(state["model"])
        model.assert_matches_accounting()
        schedule = make_batch_schedule(t_cfg["effective_batch_tokens"], t_cfg["micro_batch"],
                                       cfg.block_size)
        loss_fp32 = evaluate(model, val, schedule, t_cfg["eval_batches"], device)
        n_gains = 0
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, RMSNorm):
                    m.weight.copy_(m.weight.to(torch.bfloat16).to(torch.float32))
                    n_gains += m.weight.numel()
        if n_gains != int(r["norm_params"]):
            raise RuntimeError(f"{run_id}: rounded {n_gains} gains, results.csv has "
                               f"{r['norm_params']}")
        loss_bf16 = evaluate(model, val, schedule, t_cfg["eval_batches"], device)
        logged = float(state["history"][-1]["val"])
        rec = {
            "run_id": run_id, "sweep": r["sweep"], "budget_name": r["budget_name"],
            "bits": bits, "seed": int(r["seed"]), "step": int(state["step"]),
            "device": str(device), "train_device": r["device"],
            "val_final_logged": logged, "val_final_fp32_gains": loss_fp32,
            "val_final_bf16_gains": loss_bf16, "delta_bf16_minus_fp32": loss_bf16 - loss_fp32,
            "reproduction_error": loss_fp32 - logged,
        }
        print(json.dumps(rec))
        out.append(rec)
    if not out:
        raise SystemExit("no final-step checkpoint of a complete run found")
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out} ({len(out)} checkpoints)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
