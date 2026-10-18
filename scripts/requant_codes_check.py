#!/usr/bin/env python
"""Are the integer codes of a trained checkpoint a fixed point of re-quantization?

``isomem.quant`` recomputes the step from the weights it is given: ``compute_step(w, bits)``
is the absmean (b = 1.58) or the Gaussian MSE-optimal multiple of the standard deviation
(b >= 2) of the tensor. Quantizing the latents ``w`` gives codes ``q = quantize_codes(w, b,
step(w))`` and the dequantized tensor ``q1 = q * step(w)``. A deployment that stores ``q1``
and quantizes it again recomputes the step from ``q1``, whose statistics are those of the
rounded tensor, not of ``w``. This script measures how many codes change.

For every final-step checkpoint under ``runs/`` that belongs to a complete run of
``results.csv`` it counts, over every quantized matrix of the model (the tied embedding
included), the codes with ``quantize_codes(w, b, step(w)) != quantize_codes(q1, b, step(q1))``
and divides by the number of codes. The 16-bit control has no codes and is excluded.
Checkpoints hold the final-step weights.

Writes ``results/requant_codes_check.json``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from isomem import quant as Q  # noqa: E402
from isomem.accounting import ModelConfig  # noqa: E402
from isomem.fixed import FIXED, git_sha  # noqa: E402
from isomem.model import IsoMemGPT  # noqa: E402

RESULTS_CSV = REPO / "results" / "results.csv"
OUT = REPO / "results" / "requant_codes_check.json"


def requant_changed(w: torch.Tensor, bits: float) -> tuple[int, int]:
    """(number of codes that change on re-quantizing the dequantized tensor, number of codes)."""
    if bits == Q.BF16_CONTROL_BITS:
        raise ValueError("the 16-bit control has no integer codes")
    w = w.detach().float()
    step = Q.compute_step(w, bits)
    codes = Q.quantize_codes(w, bits, step)
    q1 = Q.quantize_to_grid(w, bits, step)
    codes2 = Q.quantize_codes(q1, bits, Q.compute_step(q1, bits))
    return int((codes != codes2).sum().item()), int(codes.numel())


def model_requant(model: IsoMemGPT) -> dict:
    changed = total = 0
    n_tensors = 0
    for _, layer in model.quantized_layers():
        c, n = requant_changed(layer.weight, model.bits)
        changed, total, n_tensors = changed + c, total + n, n_tensors + 1
    return {"n_changed": changed, "n_codes": total, "n_tensors": n_tensors,
            "share_changed": changed / total}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, default=REPO / "runs")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    rows = {r["run_id"]: r for r in csv.DictReader(open(RESULTS_CSV))}
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
        bits = float(r["bits_per_weight"])
        cfg = ModelConfig(n_layer=int(r["n_layer"]), n_embd=int(r["n_embd"]),
                          mlp_ratio=float(r["mlp_ratio"]), vocab_size=int(r["vocab"]),
                          block_size=FIXED["model"]["block_size"])
        model = IsoMemGPT(cfg, bits, rope_theta=FIXED["model"]["rope_theta"])
        model.load_state_dict(state["model"])
        model.assert_matches_accounting()
        rec = {"run_id": run_id, "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
               "budget_name": r["budget_name"], "bits": bits, "seed": int(r["seed"]),
               "step": int(state["step"]), **model_requant(model)}
        print(json.dumps(rec))
        out.append(rec)
    if not out:
        raise SystemExit("no final-step checkpoint of a complete run found")
    rep = {
        "what": "share of integer codes that change when a quantized tensor is quantized again "
                "with its recomputed step (scripts/requant_codes_check.py)",
        "torch": torch.__version__, "git_sha": git_sha(), "checkpoints": out,
    }
    args.out.write_text(json.dumps(rep, indent=2) + "\n")
    print(f"wrote {args.out} ({len(out)} checkpoints)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
