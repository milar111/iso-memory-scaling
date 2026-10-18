#!/usr/bin/env python
"""What does rounding the dequantized weights to bf16 do?

Training and every evaluation quantize the fp32 latent weight ``w`` to ``q * step`` in fp32
(``isomem.quant.quantize_to_grid``) and then, because the activations are bf16, round that
dequantized weight to bf16 where it enters each matrix product (``QuantLinear.forward``,
the tied head in ``IsoMemGPT.forward``). The weight that is applied is therefore
``bf16(q * step)``, not ``q * step``. A bf16 number has an 8-bit significand, so
``|bf16(x) - x| <= 2**-8 |x|`` and the displacement in units of the step is at most
``|q| / 256``: at most 0.5 step at b = 8 and 0.125 step at b = 6. The bound is checked here
as an assertion, not assumed.

For every final-step checkpoint under ``runs/`` that belongs to a complete run of
``results.csv``, on CPU, this script measures

* the displacement ``|bf16(q * step) - q * step| / step`` over every quantized matrix
  (including the tied embedding): its maximum, its root mean square and the share of
  weights that bf16 does not represent exactly;
* the share of weights whose integer code changes (``round(bf16(q * step) / step) != q``)
  and the number of distinct values per tensor before and after the cast;
* the squared error of the weights against their latents, before and after the cast, over
  the weights that are not clipped (``round(w / step)`` inside the grid), because clipped
  weights dominate the raw error; reported pooled in absolute units (``sq_err_rise``) and
  with each tensor's error in units of its own step (``sq_err_rise_in_steps``), because the
  two weightings differ and the rise is small enough for them to differ in sign;
* the validation loss on the fixed 50 batches (``isomem.train.evaluate``, validation split
  only) under three arms: ``trained`` (the code as run), ``exact`` (the matrix products use
  the fp32 dequantized weight with the same bf16 activations, cast to fp32 for the product;
  products and logits are rounded to bf16 as in the trained path, so the only difference is
  the weight cast) and ``replica``
  (the same code path as ``exact`` but with the weight rounded to bf16, which must
  reproduce ``trained``: it shows that the replica of the forward pass is faithful, so that
  ``exact - trained`` is the effect of the weight cast and nothing else).

The 16-bit control is excluded: its latents are cast to bf16 and there is no step to
measure against. The embedding lookup rounds each row to bf16 as an activation in all
three arms. Checkpoints hold the final-step weights, not those at ``val_loss_best``.

Writes ``results/bf16_cast_check.json``.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from isomem import data as D  # noqa: E402
from isomem import model as M  # noqa: E402
from isomem import quant as Q  # noqa: E402
from isomem.accounting import ModelConfig  # noqa: E402
from isomem.fixed import FIXED, git_sha  # noqa: E402
from isomem.model import IsoMemGPT  # noqa: E402
from isomem.train import evaluate, make_batch_schedule  # noqa: E402

RESULTS_CSV = REPO / "results" / "results.csv"
OUT = REPO / "results" / "bf16_cast_check.json"
BF16_UNIT_ROUNDOFF = 2.0 ** -8      # |bf16(x) - x| <= 2**-8 |x| (8-bit significand)
BOUND_SLACK = 1e-6                  # float32 evaluation of a bound that is exact in real numbers


# --------------------------------------------------------------------------------------
# displacement of the dequantized weights (pure, tested on synthetic tensors)
# --------------------------------------------------------------------------------------


def tensor_cast_stats(w: torch.Tensor, bits: float) -> dict:
    """Statistics of rounding ``fake_quant(w, bits)`` to bf16, for one tensor.

    Raises if the displacement exceeds the analytic bound ``|q| / 256`` steps, or if the
    cast changes an integer code or merges two levels.
    """
    if bits == Q.BF16_CONTROL_BITS:
        raise ValueError("the 16-bit control has no step to measure the cast against")
    w = w.detach().float()
    step = Q.compute_step(w, bits)
    codes = Q.quantize_codes(w, bits, step)
    deq = Q.quantize_to_grid(w, bits, step)
    if not torch.equal(deq, codes * step):
        raise AssertionError("quantize_to_grid and quantize_codes disagree")
    applied = deq.to(torch.bfloat16).float()
    disp = (applied - deq).abs() / step
    bound = codes.abs() * BF16_UNIT_ROUNDOFF
    if bool((disp > bound * (1 + BOUND_SLACK) + 1e-12).any()):
        raise AssertionError("bf16 displacement exceeds |q| / 256 of a step")
    qmin, qmax = Q.grid_range(bits)
    recoded = torch.clamp(torch.round(applied / step), qmin, qmax)
    raw = torch.round(w / step)
    unclipped = (raw >= qmin) & (raw <= qmax)
    before = torch.unique(deq).numel()
    after = torch.unique(applied).numel()
    if after != before:
        raise AssertionError(f"the cast merged or split levels: {before} -> {after}")
    err_before = ((w - deq) ** 2)[unclipped].double().sum().item()
    err_after = ((w - applied) ** 2)[unclipped].double().sum().item()
    # the same in units of the tensor's own step, so that tensors with a larger step do not
    # weigh more than those with a smaller one
    estep_before = (((w - deq) / step) ** 2)[unclipped].double().sum().item()
    estep_after = (((w - applied) / step) ** 2)[unclipped].double().sum().item()
    return {
        "n": int(w.numel()),
        "max_disp_step": float(disp.max().item()),
        "sum_sq_disp_step": float((disp.double() ** 2).sum().item()),
        "n_not_bf16_exact": int((applied != deq).sum().item()),
        "n_code_changed": int((recoded != codes).sum().item()),
        "max_abs_code": float(codes.abs().max().item()),
        "n_unclipped": int(unclipped.sum().item()),
        "sum_sq_err_before": err_before,
        "sum_sq_err_after": err_after,
        "sum_sq_err_before_steps": estep_before,
        "sum_sq_err_after_steps": estep_after,
        "distinct_before": int(before),
        "distinct_after": int(after),
    }


def merge_stats(parts: list[dict]) -> dict:
    """Combine per-tensor statistics into one record (maxima, sums, shares)."""
    if not parts:
        raise ValueError("no tensors")
    n = sum(p["n"] for p in parts)
    nu = sum(p["n_unclipped"] for p in parts)
    eb = sum(p["sum_sq_err_before"] for p in parts)
    ea = sum(p["sum_sq_err_after"] for p in parts)
    sb = sum(p["sum_sq_err_before_steps"] for p in parts)
    sa = sum(p["sum_sq_err_after_steps"] for p in parts)
    if eb <= 0:
        raise ValueError("no quantization error over the unclipped weights")
    return {
        "n_weights": n,
        "max_disp_step": max(p["max_disp_step"] for p in parts),
        "rms_disp_step": math.sqrt(sum(p["sum_sq_disp_step"] for p in parts) / n),
        "frac_not_bf16_exact": sum(p["n_not_bf16_exact"] for p in parts) / n,
        "n_code_changed": sum(p["n_code_changed"] for p in parts),
        "max_abs_code": max(p["max_abs_code"] for p in parts),
        "max_distinct_before": max(p["distinct_before"] for p in parts),
        "max_distinct_after": max(p["distinct_after"] for p in parts),
        "n_unclipped": nu,
        "frac_unclipped": nu / n,
        "sq_err_before_per_weight": eb / nu,
        "sq_err_after_per_weight": ea / nu,
        "sq_err_rise": ea / eb - 1.0,
        "sq_err_rise_in_steps": sa / sb - 1.0,
    }


def model_cast_stats(model: IsoMemGPT) -> dict:
    """:func:`merge_stats` over every quantized matrix of ``model`` (the tied embedding too)."""
    return merge_stats([tensor_cast_stats(layer.weight, model.bits)
                        for _, layer in model.quantized_layers()])


# --------------------------------------------------------------------------------------
# loss with the exact dequantized weights
# --------------------------------------------------------------------------------------


def _linear(x: torch.Tensor, w: torch.Tensor, cast: bool) -> torch.Tensor:
    """x is bf16. ``cast``: round w to bf16 (as trained); else apply the fp32 weight."""
    if cast:
        w = w.to(torch.bfloat16).float()
    return F.linear(x.float(), w).to(x.dtype)


def _quant_linear_forward(cast: bool):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _linear(x, self.quantized_weight(), cast)
    return forward


def _gpt_forward(cast: bool):
    def forward(self, idx, targets=None, dtype=torch.bfloat16):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        w = self.embed.quantized_weight()
        x = F.embedding(idx, w).to(dtype)
        cos, sin = self.rope_cos.to(x.device), self.rope_sin.to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_final(x)
        logits = _linear(x, w, cast)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
    return forward


@contextlib.contextmanager
def forward_replica(cast: bool):
    """Replace the matrix products by fp32 products of the bf16 activation with the weight
    either rounded to bf16 (``cast``, the replica of the trained path) or exact."""
    saved = (Q.QuantLinear.forward, M.IsoMemGPT.forward)
    Q.QuantLinear.forward = _quant_linear_forward(cast)
    M.IsoMemGPT.forward = _gpt_forward(cast)
    try:
        yield
    finally:
        Q.QuantLinear.forward, M.IsoMemGPT.forward = saved


def loss_arms(model: IsoMemGPT, tokens, schedule, n_batches: int, device) -> dict:
    trained = evaluate(model, tokens, schedule, n_batches, device)
    with forward_replica(cast=True):
        replica = evaluate(model, tokens, schedule, n_batches, device)
    with forward_replica(cast=False):
        exact = evaluate(model, tokens, schedule, n_batches, device)
    return {"val_trained_path": trained, "val_replica_cast": replica, "val_exact_weights": exact,
            "replica_minus_trained": replica - trained, "exact_minus_trained": exact - trained}


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------


def validate_report(rep: dict) -> None:
    """The assertions the paper's numbers rest on; raises on the first violated one."""
    cks = rep["checkpoints"]
    if not cks:
        raise AssertionError("no checkpoint")
    for c in cks:
        s = c["cast"]
        if s["n_code_changed"] != 0:
            raise AssertionError(f"{c['run_id']}: the cast changed {s['n_code_changed']} codes")
        if s["max_distinct_after"] != s["max_distinct_before"]:
            raise AssertionError(f"{c['run_id']}: distinct levels changed")
        if s["max_disp_step"] > s["max_abs_code"] * BF16_UNIT_ROUNDOFF * (1 + BOUND_SLACK):
            raise AssertionError(f"{c['run_id']}: displacement above the |q|/256 bound")
        if "loss" in c and abs(c["loss"]["replica_minus_trained"]) > rep["replica_tolerance"]:
            raise AssertionError(f"{c['run_id']}: the forward replica does not reproduce the "
                                 f"trained path ({c['loss']['replica_minus_trained']:.2e})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, default=REPO / "runs")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--device", default="cpu", help="evaluation device (default cpu)")
    ap.add_argument("--no-loss", action="store_true", help="displacement statistics only")
    ap.add_argument("--replica-tolerance", type=float, default=2e-5,
                    help="largest |replica - trained| loss accepted (nats)")
    args = ap.parse_args(argv)

    rows = {r["run_id"]: r for r in csv.DictReader(open(RESULTS_CSV))}
    device = torch.device(args.device)
    t_cfg = FIXED["training"]
    val = None if args.no_loss else D.load_tokens("val")   # validation split only

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
        model = IsoMemGPT(cfg, bits, rope_theta=FIXED["model"]["rope_theta"]).to(device)
        model.load_state_dict(state["model"])
        model.assert_matches_accounting()
        rec = {"run_id": run_id, "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
               "budget_name": r["budget_name"], "bits": bits,
               "seed": int(r["seed"]), "step": int(state["step"]),
               "cast": model_cast_stats(model)}
        if val is not None:
            schedule = make_batch_schedule(t_cfg["effective_batch_tokens"],
                                           t_cfg["micro_batch"], cfg.block_size)
            rec["loss"] = loss_arms(model, val, schedule, t_cfg["eval_batches"], device)
            rec["val_final_logged"] = float(state["history"][-1]["val"])
        print(json.dumps(rec))
        out.append(rec)
    if not out:
        raise SystemExit("no final-step checkpoint of a complete run found")
    rep = {
        "what": "bf16 rounding of the dequantized weights (scripts/bf16_cast_check.py)",
        "device": str(device), "torch": torch.__version__, "git_sha": git_sha(),
        "split": None if val is None else "val",
        "n_eval_batches": None if val is None else int(t_cfg["eval_batches"]),
        "replica_tolerance": args.replica_tolerance,
        "checkpoints": out,
    }
    validate_report(rep)
    args.out.write_text(json.dumps(rep, indent=2) + "\n")
    print(f"wrote {args.out} ({len(out)} checkpoints)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
