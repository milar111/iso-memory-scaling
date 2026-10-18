"""Share of latent weights the quantizer clips, per primary run (diagnostic, CPU).

For every complete PRIMARY-sweep TinyStories run with ``b < 16``, its final-step checkpoint
is loaded and, for every quantized matrix (attention qkv/proj, MLP up/down, the tied
embedding/head -- counted once, it is one matrix), the fraction of latent fp32 weights
that the forward pass's quantizer clips is counted.

Definition (exact, not approximate): with ``step = isomem.quant.compute_step(w, b)`` (the
absmean rule; ternary is BitNet's ``mean|w|``) and the code range
``[qmin, qmax] = isomem.quant.grid_range(b)``, a weight is clipped iff its unclamped code
``round(w / step)`` lies outside ``[qmin, qmax]`` -- i.e. iff the clamp in
``quantize_to_grid`` changes it. Equivalently ``w / step >= qmax + 0.5`` or
``w / step < qmin - 0.5`` (up to torch.round's half-to-even ties). The grid is asymmetric
for ``b >= 2`` (``qmin = -2**(b-1)``, ``qmax = 2**(b-1) - 1``), so the positive and negative
thresholds differ; ternary is symmetric at ``1.5 * mean|w|``. Every layer's recomputed
dequantized weight is asserted bitwise equal to ``QuantLinear.quantized_weight()``, so the
count is of the forward pass's own quantizer, not of a re-implementation of it.

A run's share is element-weighted over the whole model: total clipped / total quantized
weights. ``b = 16`` (the bf16 control) has no quantizer and therefore no clipping; those
runs are EXCLUDED (not reported as 0), and their count is recorded as ``n_excluded_b16``.

Code entropy (same pass, same runs). For every quantized matrix the integer codes
``clamp(round(w / step), qmin, qmax)`` -- asserted to be the forward pass's own, as above --
are histogrammed over the matrix's ``qmax - qmin + 1`` levels, and the matrix's empirical
Shannon entropy ``H = -sum_k p_k log2 p_k`` (bits per weight) is taken. A run's ``entropy``
is the element-weighted mean of the per-matrix entropies over every quantized matrix (tied
embedding once). It is the size a per-matrix order-0 entropy coder would approach, against
the fixed-width ``log2(levels)`` the budget charges. ``entropy_by_bits`` is the unweighted
mean over runs. ``b = 16`` is EXCLUDED here too: it has no integer codes.

Gaussian reference (analytic, no checkpoint). ``gaussian_share_by_bits[b]`` is the share a
unit Gaussian N(0, 1) would have clipped under the same step rule, with the step taken from
the population absmean ``sqrt(2/pi)`` through ``isomem.quant.compute_step`` itself (so its
tabulated loading factor): ``P(x >= (qmax + 1/2) step) + P(x < (qmin - 1/2) step)``. The
observed share minus it is the excess the trained weights' tails produce.

Usage (on the compute host, where every primary checkpoint is):

    .venv/bin/python scripts/clip_share.py                       # -> results/clip_share.json
    .venv/bin/python scripts/clip_share.py --ckpt-dir ~/bit-budget/runs
    # local measurement on the few checkpoints present here (never the default output):
    .venv/bin/python scripts/clip_share.py --only B847872-b6-s0 --out /tmp/x.json

Reads no data split at all: only checkpoints and results.csv.
"""

from __future__ import annotations

import argparse
import math
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from isomem.analysis import load_rows  # noqa: E402
from isomem.corpus import ACTIVE as CORPUS  # noqa: E402
from isomem.provenance import CheckpointProvenanceError  # noqa: E402
from isomem.fixed import FIXED, assert_fixed_invariants, git_sha  # noqa: E402
from isomem.model import IsoMemGPT  # noqa: E402
from isomem.quant import (  # noqa: E402
    BF16_CONTROL_BITS, compute_step, grid_range, quantize_codes)


def _load_a7():
    """scripts/a7_average.py, for its run reconstruction (the same one M1 used on every run)."""
    name = "a7_average"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "a7_average.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A7 = _load_a7()
reconstruct = A7.reconstruct

RESULTS_CSV = CORPUS.results_csv
DEFAULT_CKPT_DIR = REPO_ROOT / "runs"
DEFAULT_OUT = REPO_ROOT / "results" / "clip_share.json"


def bits_key(bits: float) -> str:
    return f"{bits:g}"


@torch.no_grad()
def layer_clip_counts(layer) -> tuple[int, int, int]:
    """(n_clipped_high, n_clipped_low, n_total) for one QuantLinear, by its own quantizer."""
    bits = layer.bits
    if bits == BF16_CONTROL_BITS:
        raise ValueError("b = 16 has no quantizer and no clipping threshold")
    w = layer.weight.detach()
    step = compute_step(w, bits)
    step_t = torch.as_tensor(step, dtype=w.dtype, device=w.device)
    raw = torch.round(w / step_t)
    qmin, qmax = grid_range(bits)
    deq = torch.clamp(raw, qmin, qmax) * step_t
    if not torch.equal(deq, layer.quantized_weight()):
        raise AssertionError("recomputed quantization differs from the forward pass's")
    return int((raw > qmax).sum()), int((raw < qmin).sum()), w.numel()


@torch.no_grad()
def code_entropy(codes: torch.Tensor, bits: float) -> float:
    """Empirical Shannon entropy, bits per weight, of one matrix's integer codes."""
    qmin, qmax = grid_range(bits)
    c = codes.detach().reshape(-1).to(torch.int64)
    if c.numel() == 0:
        raise ValueError("no codes")
    if int(c.min()) < qmin or int(c.max()) > qmax:
        raise ValueError(f"codes outside [{qmin}, {qmax}]")
    counts = torch.bincount(c - qmin, minlength=qmax - qmin + 1).to(torch.float64)
    p = counts[counts > 0] / c.numel()
    return float(-(p * torch.log2(p)).sum()) + 0.0      # + 0.0: no -0.0 for a single level


@torch.no_grad()
def layer_code_entropy(layer) -> tuple[float, int]:
    """(entropy of the layer's integer codes in bits per weight, n_weights), by its own
    quantizer: codes * step is asserted bitwise equal to QuantLinear.quantized_weight()."""
    bits = layer.bits
    if bits == BF16_CONTROL_BITS:
        raise ValueError("b = 16 has no quantizer and no integer codes")
    w = layer.weight.detach()
    step = compute_step(w, bits)
    codes = quantize_codes(w, bits, step)
    if not torch.equal(codes * torch.as_tensor(step, dtype=w.dtype, device=w.device),
                       layer.quantized_weight()):
        raise AssertionError("recomputed codes differ from the forward pass's")
    return code_entropy(codes, bits), w.numel()


def gaussian_clip_share(bits: float) -> float:
    """Share of a unit Gaussian the step rule clips at ``bits`` (analytic).

    The step is isomem.quant.compute_step applied to a tensor whose absmean is the unit
    Gaussian's population absmean sqrt(2/pi), so the quantizer's own constants set it."""
    if bits == BF16_CONTROL_BITS:
        raise ValueError("b = 16 has no quantizer and no clipping threshold")
    mu = math.sqrt(2.0 / math.pi)
    step = float(compute_step(torch.tensor([mu], dtype=torch.float64), bits))
    qmin, qmax = grid_range(bits)
    upper = 0.5 * math.erfc((qmax + 0.5) * step / math.sqrt(2.0))   # P(x >= (qmax+.5) step)
    lower = 0.5 * math.erfc(-(qmin - 0.5) * step / math.sqrt(2.0))  # P(x <  (qmin-.5) step)
    return upper + lower


def layer_kind(name: str) -> str:
    if name == "embed":
        return "embedding"
    if ".attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    raise AssertionError(f"unexpected quantized layer {name!r}")


def model_clip_share(model: IsoMemGPT) -> dict:
    """Element-weighted clip share over every quantized matrix, plus a per-kind breakdown."""
    tot_c = tot_n = tot_hi = 0
    ent_weighted = 0.0
    kinds: dict[str, list[int]] = {}
    layers = model.quantized_layers()
    if not layers:
        raise AssertionError("model has no quantized layers")
    for name, layer in layers:
        hi, lo, n = layer_clip_counts(layer)
        h, n_h = layer_code_entropy(layer)
        if n_h != n:
            raise AssertionError(f"{name}: entropy and clip counts cover different weights")
        ent_weighted += h * n
        tot_c += hi + lo
        tot_hi += hi
        tot_n += n
        k = kinds.setdefault(layer_kind(name), [0, 0])
        k[0] += hi + lo
        k[1] += n
    return {
        "share": tot_c / tot_n,
        "n_clipped": tot_c,
        "n_clipped_positive": tot_hi,
        "n_weights": tot_n,
        "share_by_kind": {k: c / n for k, (c, n) in sorted(kinds.items())},
        "entropy": ent_weighted / tot_n,
    }


def target_rows(results_csv: Path, only: list[str] | None) -> tuple[list[dict], int]:
    """Complete primary rows with b < 16 (deduped, last write wins), and the b = 16 count."""
    rows = [r for r in load_rows(results_csv)
            if r["sweep"] == "primary" and r["status"] == "complete"]
    if only:
        missing = set(only) - {r["run_id"] for r in rows}
        if missing:
            raise KeyError(f"not complete primary runs: {sorted(missing)}")
        rows = [r for r in rows if r["run_id"] in only]
    n16 = sum(1 for r in rows if float(r["bits_per_weight"]) == BF16_CONTROL_BITS)
    return [r for r in rows if float(r["bits_per_weight"]) != BF16_CONTROL_BITS], n16


def load_final_model(row: dict, ckpt_dir: Path) -> IsoMemGPT:
    """The run's final-step checkpoint into a model rebuilt from its replayed configuration."""
    setup = reconstruct(row)
    path = Path(ckpt_dir) / f"{setup.run_id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path}: checkpoint of a complete primary run is missing")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["step"] != setup.max_steps:
        raise CheckpointProvenanceError(
            f"{setup.run_id}: checkpoint at step {ck['step']}, the run has {setup.max_steps}")
    model = IsoMemGPT(setup.cfg, setup.bits, rope_theta=FIXED["model"]["rope_theta"])
    model.assert_matches_accounting()
    model.load_state_dict(ck["model"])          # strict: shapes must match the config
    return model


def summarise(per_run: list[dict]) -> tuple[dict, dict, dict]:
    by_bits: dict[str, list[float]] = {}
    by_bb: dict[str, dict[str, list[float]]] = {}
    for r in per_run:
        k = bits_key(r["b"])
        by_bits.setdefault(k, []).append(r["share"])
        by_bb.setdefault(r["budget"], {}).setdefault(k, []).append(r["share"])
    order = lambda d: dict(sorted(d.items(), key=lambda kv: float(kv[0])))  # noqa: E731
    mean_bits = order({k: statistics.mean(v) for k, v in by_bits.items()})
    mean_bb = {bud: order({k: statistics.mean(v) for k, v in d.items()})
               for bud, d in by_bb.items()}
    sd_bb = {bud: order({k: (statistics.stdev(v) if len(v) > 1 else None)
                         for k, v in d.items()}) for bud, d in by_bb.items()}
    return mean_bits, mean_bb, sd_bb


def write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def compute(results_csv: Path, ckpt_dir: Path, only: list[str] | None = None) -> dict:
    rows, n16 = target_rows(results_csv, only)
    per_run = []
    for row in rows:
        t0 = time.time()
        m = model_clip_share(load_final_model(row, ckpt_dir))
        per_run.append({"run_id": row["run_id"], "b": float(row["bits_per_weight"]),
                        "budget": row["budget_name"], "seed": int(row["seed"]),
                        **m, "seconds": time.time() - t0})
        print(f"  {row['run_id']:>22}  share {m['share']:.6f}  H {m['entropy']:.4f}  "
              f"({m['n_clipped']}/{m['n_weights']})  {time.time() - t0:.2f}s", flush=True)
    by_bits, by_bb, sd_bb = summarise(per_run)
    ent: dict[str, list[float]] = {}
    for r in per_run:
        ent.setdefault(bits_key(r["b"]), []).append(r["entropy"])
    entropy_by_bits = {k: statistics.mean(ent[k]) for k in by_bits}
    gauss = {k: gaussian_clip_share(float(k)) for k in by_bits}
    return {
        "by_bits": by_bits,
        "entropy_by_bits": entropy_by_bits,
        "gaussian_share_by_bits": gauss,
        "by_bits_budget": by_bb,
        "by_bits_budget_sd": sd_bb,
        "per_run": per_run,
        "git_sha": git_sha(),
        "n_runs": len(per_run),
        "n_excluded_b16": n16,
        "partial": bool(only),
        "definition": (
            "share of latent weights whose unclamped code round(w/step) lies outside "
            "grid_range(b), step = isomem.quant.compute_step (absmean rule; ternary mean|w|); "
            "element-weighted over all quantized matrices (attention, MLP, tied embedding "
            "once); final-step checkpoint; b=16 excluded (no quantizer)"),
        "by_bits_is": "unweighted mean over runs of the per-run share, all budgets pooled",
        "entropy_definition": (
            "per run: element-weighted mean over all quantized matrices (tied embedding once) "
            "of each matrix's empirical Shannon entropy, bits per weight, of its integer codes "
            "clamp(round(w/step), grid_range(b)); entropy_by_bits: unweighted mean over runs, "
            "budgets pooled; b=16 excluded (no codes)"),
        "gaussian_definition": (
            "share of a unit Gaussian clipped by the same rule, step = compute_step at the "
            "population absmean sqrt(2/pi): P(x >= (qmax+1/2) step) + P(x < (qmin-1/2) step); "
            "analytic"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    ap.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", action="append", default=None,
                    help="restrict to these run ids (measurement only; needs a non-default --out)")
    args = ap.parse_args(argv)

    assert_fixed_invariants()
    if not CORPUS.is_default:
        raise SystemExit("clip_share is defined on the TinyStories primary sweep; "
                         "unset ISOMEM_DATASET")
    if args.only and args.out.resolve() == DEFAULT_OUT.resolve():
        raise SystemExit("--only gives a partial result; write it somewhere other than "
                         f"{DEFAULT_OUT}")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"device cpu   git {git_sha()}   ckpts {args.ckpt_dir}", flush=True)
    out = compute(args.results_csv, args.ckpt_dir, args.only)
    write_json_atomic(args.out, out)
    print("by b (mean over runs): "
          + "  ".join(f"b={k}: {v:.6f}" for k, v in out["by_bits"].items()))
    print("gaussian reference:    "
          + "  ".join(f"b={k}: {v:.6f}" for k, v in out["gaussian_share_by_bits"].items()))
    print("code entropy (bits):   "
          + "  ".join(f"b={k}: {v:.4f}" for k, v in out["entropy_by_bits"].items()))
    print(f"{out['n_runs']} runs (b=16 excluded: {out['n_excluded_b16']}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
