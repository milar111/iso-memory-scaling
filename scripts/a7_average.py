#!/usr/bin/env python
"""A7 M1: the averaging continuation of every complete TinyStories run.

    cd ~/bit-budget-a7 && python scripts/a7_average.py \
        --main-dir ~/bit-budget --ckpt-dir ~/bit-budget/runs      (normally via a7_tick.sh)

It refuses to run from, or write into, the main checkout (--main-dir).

PREREGISTRATION.md §16 (Amendment A7), subsection "M1 -- averaging continuation", is the
contract; this script does what it says and nothing more. For each of the 142 complete
TinyStories runs (primary and robustness), starting from the run's final checkpoint with the
model AND the AdamW state restored:

* the checkpoint is evaluated as saved with the standard evaluation (``train.evaluate``: 50
  batches of 128 x 256 tokens, ``EVAL_SEED``). If that differs from the run's recorded final
  validation loss by more than 1e-3 the run is recorded as ``setup-mismatch`` and not
  continued. It is also evaluated on 50 fresh validation batches drawn with
  ``FRESH_EVAL_SEED`` (``val_fresh``), which measures the selection bias of the fixed set;
* continuation 1: ``K`` optimizer steps at the schedule's final learning rate
  (``lr * min_lr_frac``) on the run's own batch stream, continuing from its final micro-step,
  with the run's weight decay, gradient clip and micro-batch. Every ``EVAL_EVERY`` steps the
  last iterate is evaluated (a): ``A`` is the mean of those 16 values. At every step the
  integer code of every quantized weight (b < 16) and every fp32 parameter are accumulated;
* (b) the time-averaged code, rounded, times the per-tensor step of the FINAL latent weights
  -> ``val_b``. (c) the time-averaged fp32 parameters, quantized by the forward pass as in
  training -> ``val_c``, the corrected metric;
* continuation 2, from a fresh reload of the same checkpoint: ``K`` steps, step ``t`` at
  ``lr * min_lr_frac * (1 - t/K)``: ``K`` non-zero updates, reaching 0 at the end of the last
  step -> ``val_d`` (PREREGISTRATION.md §17);
* only then are the (c) weights saved, to ``<out-dir>/averaged/<run_id>.pt``, for the single
  test read (§14; the location is §17's), and the row appended.

One row per run goes to ``<out-dir>/m1.csv``, only once the run is fully done; a run already
there is skipped, and an interrupted run simply reruns. Nothing is written under the main
checkout (``--main-dir``, which holds the checkpoints) or to ``results/results.csv``: an
audit hook refuses every Python-level write there while the script runs, and
a test (not part of this release) checks the files' hashes.

It runs on the compute host beside the FineWeb-Edu arm, from a separate directory, so it
shares the GPU. Before each run it checks quiet hours (no run starts within 20 min of
SLEEP_AT: sweep.next_deadline's 10-min margin plus QUIET_MARGIN_S) and GPU memory. While the
arm is not finalised, the arm's worst-case footprint (``--arm-worst-gib``) is kept free for
it: this process's caching allocator is capped (set_per_process_memory_fraction) to what is
left, so it cannot take the arm's memory whatever a run needs. A run that does not fit OOMs
inside that cap, ends the invocation, and while the arm is live is listed in
``oom_deferred.txt`` to wait for the arm instead of being retried. A setup mismatch, NaN or collapse makes the invocation exit 1, and so does any
recorded row that is not ``complete``, so the tick never retires over a failure.

Exit codes: 0 every target run is recorded complete; 3 work remains (quiet hours, GPU memory,
an OOM, another invocation holding the lock, or ``--limit``/``--only``); 1 an error.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# The copy of isomem next to this script, never an editable install elsewhere: on the host
# the interpreter is the main checkout's venv, whose isomem is the FineWeb arm's code.
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import isomem  # noqa: E402
from isomem import data as D  # noqa: E402
from isomem import train as TR  # noqa: E402
from isomem.accounting import ModelConfig, assert_within_budget, count_params  # noqa: E402
from isomem.analysis import load_rows  # noqa: E402
from isomem.corpus import ACTIVE as CORPUS  # noqa: E402
from isomem.provenance import CheckpointProvenanceError  # noqa: E402
from isomem.fixed import FIXED, assert_fixed_invariants, git_sha, learning_rate  # noqa: E402
from isomem.model import IsoMemGPT  # noqa: E402
from isomem.quant import (  # noqa: E402
    BF16_CONTROL_BITS,
    TernaryCollapse,
    compute_step,
    grid_range,
    quantize_codes,
)
from isomem.search import search_cell  # noqa: E402
from isomem.sweep import (  # noqa: E402
    _quiet_hours_values,
    next_deadline,
    quiet_hours_settings,
    run_id as sweep_run_id,
    token_budget,
)

# --- fixed by A7 ----------------------------------------------------------------------------
K_DEFAULT = 400
EVAL_EVERY = 25                          # 400 / 25 = 16 evaluations of the last iterate
EVAL_BATCHES = FIXED["training"]["eval_batches"]     # the standard evaluation: 50 batches
N_FRESH_BATCHES = 50
# The fresh-batch seed. Any value other than train.EVAL_SEED (987_654_321) draws a different
# set of validation windows; this one is the date A7 was recorded, fixed before any M1 run.
FRESH_EVAL_SEED = 20_260_927
MISMATCH_TOL = 1e-3
SWEEPS = ("primary", "robustness")
EXPECTED_RUNS = 142                      # PREREGISTRATION.md §16: "all 142 complete runs"
# Added to next_deadline's own 10-min margin: no run starts within 20 min of SLEEP_AT.
QUIET_MARGIN_S = 10 * 60

# --- sharing the GPU with the FineWeb-Edu arm -----------------------------------------------
# The arm's finalised marker (sweep_tick.sh), relative to the main checkout. Only stat'ed.
ARM_MARKER = Path("results") / "fineweb_edu" / ".finalised"
# ASSUMED, not measured: the arm process's worst GPU footprint (context included). The largest
# recorded training peak of a cell the arm also trains (results.csv peak_mem_mb, 4 Mbit b=2,
# 3272.08 MB of max_memory_allocated) x 1.25 for allocator slack, + 0.5 GiB of CUDA context.
# Replace with the arm's footprint measured on the host at its 4 Mbit b=2 cell (the card's
# total minus cudaMemGetInfo's free memory, from a probe; nvidia-smi fails there while a
# driver update awaits a reboot), passed as --arm-worst-gib (a7_tick.sh: A7_ARM_WORST_GIB).
ARM_WORST_GIB_ASSUMED = (3272.078336e6 * 1.25 + 0.5 * 2**30) / 2**30
# Used only when nvidia-smi cannot report this process: its non-allocator GPU memory (CUDA
# context, loaded kernels, cuBLAS). MEASURED on the host 2026-09-27 with two overlapping
# probes, one running bf16 matmuls, SDPA, backward and AdamW: 0.151 GiB beyond the caching
# allocator's reserved memory, identical in two trials (a bare context: 0.112 GiB, 3 trials).
# Rounded up to 0.25 for kernels that probe did not load.
OWN_NONALLOC_GIB_ASSUMED = 0.25
# No run is expected to fit in less than this under the cap (a 256 Kbit continuation's
# micro-batch activations alone); below it the invocation stops instead of trying runs that
# can only OOM. ASSUMED; the first rows' peak_reserved_gib check it.
MIN_ALLOWED_GIB = 0.25
# Runs that hit OOM inside this process's cap while the arm was live: skipped until the arm
# is finalised, so that a tick never retries the same run into the same cap.
OOM_DEFERRED = "oom_deferred.txt"

RESULTS_CSV = CORPUS.results_csv
EXIT_OK, EXIT_ERROR, EXIT_MORE = 0, 1, 3
LOCK_NAME = ".m1.lock"   # in --out-dir; a7_tick.sh's own lock is a different file

FIELDS = [
    "run_id", "sweep", "budget_name", "bits", "seed", "n_layer", "n_embd", "status",
    "val_recorded_final", "val_ckpt", "val_fresh", "val_loss_best",
    "A_mean", "A_sd", "A_evals", "val_b", "val_c", "val_d",
    "k", "lr_used", "device", "peak_mem_gib", "peak_reserved_gib", "wall_s", "git_sha",
]


# --------------------------------------------------------------------------------------------
# Refusing writes into the main checkout, and reads of the test split
# --------------------------------------------------------------------------------------------


class ForbiddenAccess(PermissionError):
    """This process tried to write where it must only read, or to read the test split."""


_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_ACTIVE_GUARD: "AccessGuard | None" = None
_HOOK_INSTALLED = False


def _forms(path) -> tuple[Path, Path]:
    p = os.fsdecode(path)
    return Path(os.path.abspath(p)), Path(os.path.realpath(p))


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


class AccessGuard:
    """Active while `main` runs. Enforced by a `sys.addaudithook` hook, so it sees every
    Python-level open, rename and delete. C-level writers that bypass Python's ``open``,
    such as ``torch.save(path)``, are NOT seen: those are covered by main's refusal of any
    --out-dir that overlaps the main checkout, and by ``save_averaged`` checking its target.

    Writes are refused anywhere under the main checkout (which holds the checkpoints) and
    under the checkpoint directory, and to this copy's results.csv. Reads of the test split
    are refused under this copy's data/ and under the main checkout's data/.
    """

    def __init__(self, main_dir: Path, ckpt_dir: Path, results_csv: Path):
        self.no_write = list(dict.fromkeys(f for d in (main_dir, ckpt_dir) for f in _forms(d)))
        self.no_write_files = set(_forms(results_csv))
        self.data_roots = list(dict.fromkeys(
            f for d in (D.DATA_DIR, Path(main_dir) / "data") for f in _forms(d)))

    def check_write(self, path) -> None:
        for p in _forms(path):
            if p in self.no_write_files or any(_under(p, r) for r in self.no_write):
                raise ForbiddenAccess(f"A7 M1 must never write {p}")

    def check_read(self, path) -> None:
        for p in _forms(path):
            for root in self.data_roots:
                if _under(p, root) and "test" in str(p.relative_to(root)):
                    raise ForbiddenAccess(f"A7 M1 must never read the test split ({p})")

    def __call__(self, event: str, args: tuple) -> None:
        if event == "open":
            path, mode, flags = args
            if isinstance(path, int):
                return
            writing = (flags & _WRITE_FLAGS) if isinstance(flags, int) else \
                any(c in (mode or "") for c in "wax+")
            if writing:
                self.check_write(path)
            self.check_read(path)
        elif event in ("os.rename", "os.link", "os.symlink"):
            self.check_write(args[0])
            self.check_write(args[1])
        elif event in ("shutil.move", "shutil.copyfile"):
            # (src, dst): copying FROM a checkpoint is a read; only the destination is a
            # write. A move's removal of its source is seen as os.rename / os.remove.
            self.check_write(args[1])
        elif event in ("os.remove", "os.rmdir", "os.mkdir", "os.truncate", "os.chmod",
                       "os.utime", "shutil.rmtree"):
            if not isinstance(args[0], int):
                self.check_write(args[0])


def _audit(event: str, args: tuple) -> None:
    guard = _ACTIVE_GUARD
    if guard is not None:
        guard(event, args)


def _activate(guard: AccessGuard | None) -> None:
    global _ACTIVE_GUARD, _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit)  # cannot be removed; inert while no guard is active
        _HOOK_INSTALLED = True
    _ACTIVE_GUARD = guard


# --------------------------------------------------------------------------------------------
# Reconstructing a run exactly as trained
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Setup:
    run_id: str
    cfg: ModelConfig
    bits: float
    budget_bits: int
    seed: int
    schedule: TR.BatchSchedule
    max_steps: int
    base_lr: float


def reconstruct(row: dict) -> Setup:
    """The run's configuration from the sweep's own search, cross-checked against its row.

    Mirrors sweep.run_cell (search_cell with the cell's shape rule; max_tokens from
    token_budget) and train.train_one (budget assertion, batch schedule, max_steps, the lr
    rule). Any disagreement is a provenance failure and raises: it is a code error, never
    something to resolve by preferring one source.
    """
    bits = float(row["bits_per_weight"])
    budget_bits = int(row["budget_bits_target"])
    seed = int(row["seed"])
    rid = row["run_id"]
    if sweep_run_id(budget_bits, bits, seed, row["shape_rule"]) != rid:
        raise CheckpointProvenanceError(f"{rid}: run id does not replay from its row")

    found = search_cell(budget_bits, bits, shape_rule=row["shape_rule"])
    cfg = found.config
    recorded = (int(row["n_layer"]), int(row["n_embd"]), float(row["mlp_ratio"]),
                int(row["vocab"]), int(row["budget_bits_actual"]))
    replayed = (cfg.n_layer, cfg.n_embd, cfg.mlp_ratio, cfg.vocab_size, found.actual_bits)
    if recorded != replayed:
        raise CheckpointProvenanceError(f"{rid}: search replays to {replayed}, "
                                        f"results.csv recorded {recorded}")
    assert_within_budget(cfg, bits, budget_bits)

    t_cfg = FIXED["training"]
    schedule = TR.make_batch_schedule(t_cfg["effective_batch_tokens"],
                                      int(row["micro_batch"]), cfg.block_size)
    if (schedule.grad_accum, schedule.effective_batch_tokens) != (
            int(row["grad_accum"]), int(row["effective_batch_tokens"])):
        raise CheckpointProvenanceError(f"{rid}: batch schedule {schedule} differs from row")
    max_steps = max(1, token_budget(count_params(cfg)) // schedule.effective_batch_tokens)
    if max_steps != int(row["steps"]):
        raise CheckpointProvenanceError(
            f"{rid}: planned {max_steps} steps, row says {row['steps']}")
    base_lr = learning_rate(cfg.n_embd)
    if not math.isclose(base_lr, float(row["lr"]), rel_tol=1e-12):
        raise CheckpointProvenanceError(f"{rid}: lr rule gives {base_lr}, row {row['lr']}")
    return Setup(rid, cfg, bits, budget_bits, seed, schedule, max_steps, base_lr)


def build_model(setup: Setup, device: torch.device) -> IsoMemGPT:
    """train.py:321, and the accounting assertion that follows it."""
    model = IsoMemGPT(setup.cfg, setup.bits, rope_theta=FIXED["model"]["rope_theta"]).to(device)
    model.assert_matches_accounting()
    return model


def load_checkpoint(setup: Setup, ckpt_path: Path, device: torch.device):
    """Model and AdamW state exactly as train_one resumes them (train.py:334-352)."""
    o_cfg = FIXED["optimizer"]
    model = build_model(setup, device)
    opt = torch.optim.AdamW(
        model.param_groups(o_cfg["weight_decay"]),
        lr=setup.base_lr,
        betas=(o_cfg["beta1"], o_cfg["beta2"]),
    )
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    if ck["step"] != setup.max_steps:
        raise CheckpointProvenanceError(
            f"{setup.run_id}: checkpoint is at step {ck['step']}, the run has "
            f"{setup.max_steps}; it is not the final checkpoint")
    return model, opt, ck


# --------------------------------------------------------------------------------------------
# Learning rates
# --------------------------------------------------------------------------------------------


def final_lr(setup: Setup) -> float:
    """train.py's schedule at the final step, which is exactly lr * min_lr_frac.

    ``lr_at`` clamps progress to 1, so at ``step >= max_steps`` it returns ``min_lr`` for
    every continuation step. Asserted, not assumed.
    """
    lr = TR.lr_at(setup.max_steps, setup.max_steps, setup.base_lr)
    expected = setup.base_lr * FIXED["lr"]["min_lr_frac"]
    if lr != expected:
        raise AssertionError(f"schedule end {lr!r} != base_lr * min_lr_frac {expected!r}")
    return lr


def anneal_lr(t: int, k: int, lr_start: float) -> float:
    """Continuation 2's step ``t`` of ``k``: ``lr_start`` at the first step, falling linearly
    by ``lr_start / k`` per step, so the schedule reaches exactly 0 at the END of step ``k``.

    Not ``1 - t/(k-1)``: that puts the last step at lr 0, and AdamW at lr 0 changes no weight
    (both the decoupled decay and the Adam update scale by lr; measured, and tested), so (d)
    would make ``k - 1`` updates, not the ``k`` A7 states. PREREGISTRATION.md §17.
    """
    if k < 1 or not 0 <= t < k:
        raise ValueError(f"step {t} outside an anneal of {k} steps")
    return lr_start * (1.0 - t / k)


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------


@torch.no_grad()
def evaluate_seeded(model: IsoMemGPT, tokens, schedule: TR.BatchSchedule, n_batches: int,
                    device: torch.device, seed: int) -> float:
    """train.evaluate with the batch seed as a parameter. Identical at seed=EVAL_SEED
    (tested), so val_fresh differs from val_ckpt only in which batches it sees."""
    was_training = model.training
    model.eval()
    total = 0.0
    for i in range(n_batches):
        x, y = TR.get_batch(tokens, schedule, seed, i, device)
        _, loss = model(x, y)
        total += loss.item()
    model.train(was_training)
    return total / n_batches


def standard_eval(model: IsoMemGPT, val_tokens, setup: Setup, device) -> float:
    return TR.evaluate(model, val_tokens, setup.schedule, EVAL_BATCHES, device)


# --------------------------------------------------------------------------------------------
# Averaging
# --------------------------------------------------------------------------------------------


class RunningMean:
    """Mean of a sequence of same-shaped tensor dicts, accumulated in fp64.

    fp64 holds the sum of up to 2**29 fp32 values of similar magnitude without rounding, so
    the mean is the fp32 rounding of the exact mean for any K this script uses.
    """

    def __init__(self):
        self.sums: dict[str, torch.Tensor] = {}
        self.n = 0

    def add(self, tensors: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            if not self.sums:
                self.sums = {k: v.detach().to(torch.float64).clone() for k, v in tensors.items()}
            else:
                for k, v in tensors.items():
                    self.sums[k].add_(v.detach().to(torch.float64))
        self.n += 1

    def mean(self, dtype=torch.float32) -> dict[str, torch.Tensor]:
        if self.n == 0:
            raise ValueError("no values accumulated")
        return {k: (s / self.n).to(dtype) for k, s in self.sums.items()}


def layer_codes(model: IsoMemGPT) -> dict[str, torch.Tensor]:
    """Integer code of every quantized weight, with the repo's own quantizer (quant.py)."""
    bits = model.bits
    out = {}
    with torch.no_grad():
        for name, layer in model.quantized_layers():
            w = layer.weight.detach()
            out[name] = quantize_codes(w, bits, compute_step(w, bits))
    return out


def mean_code_weights(code_mean: dict[str, torch.Tensor], final_latents: dict[str, torch.Tensor],
                      bits: float) -> dict[str, torch.Tensor]:
    """(b): round the time-averaged code; scale by the step of the FINAL latent weights.

    ``torch.round`` rounds half to even, the same rounding quant.quantize_codes applies.
    """
    qmin, qmax = grid_range(bits)
    out = {}
    for name, mean in code_mean.items():
        codes = torch.round(mean.to(torch.float64))
        if codes.min() < qmin or codes.max() > qmax:
            raise AssertionError(f"{name}: averaged code outside the grid [{qmin}, {qmax}]")
        latent = final_latents[name]
        step = compute_step(latent, bits)
        out[name] = codes.to(latent.dtype) * step
    return out


def evaluate_fixed_quantized(setup: Setup, state: dict, qweights: dict[str, torch.Tensor],
                             val_tokens, device) -> float:
    """Evaluate with every quantized layer's forward weight replaced by ``qweights``.

    The norm gains are those of ``state``. The replacement shadows
    ``QuantLinear.quantized_weight`` on this one throwaway model, which is what both the
    tied embedding/head (model.py:175) and every linear layer call in the forward pass.
    """
    model = build_model(setup, device)
    model.load_state_dict(state)
    layers = dict(model.quantized_layers())
    if set(layers) != set(qweights):
        raise AssertionError("fixed weights do not cover exactly the quantized layers")
    for name, layer in layers.items():
        q = qweights[name].to(device)
        layer.quantized_weight = (lambda q=q: q)
    return standard_eval(model, val_tokens, setup, device)


def evaluate_state(setup: Setup, state: dict, val_tokens, device) -> float:
    """Evaluate a latent state; the forward quantizes it exactly as in training."""
    model = build_model(setup, device)
    model.load_state_dict(state)
    return standard_eval(model, val_tokens, setup, device)


# --------------------------------------------------------------------------------------------
# The continuation
# --------------------------------------------------------------------------------------------


@dataclass
class Continuation:
    evals: list[float]
    lrs: list[float]
    params: RunningMean | None
    codes: RunningMean | None
    micro_steps: list[int]


def continue_run(model: IsoMemGPT, opt, setup: Setup, train_tokens, val_tokens,
                 device: torch.device, start_step: int, k: int, lr_fn,
                 eval_every: int | None = None, track: bool = False) -> Continuation:
    """``k`` optimizer steps continuing the run's own loop (train.py:395-432).

    Step ``start_step + t`` uses ``lr_fn(t)``, fetches micro-steps
    ``step * grad_accum + micro`` of the run's batch stream, scales the loss by
    ``1/grad_accum``, clips at the fixed norm, and steps AdamW with the run's param groups
    (weight decay on matrices only). bf16 activations come from the model's forward
    (model.py:169), as in training; train.py uses no autocast. Non-finite loss or gradient
    raises NanLoss; the collapse check runs on train.py's cadence.
    """
    o_cfg, t_cfg = FIXED["optimizer"], FIXED["training"]
    collapse_every = t_cfg["collapse_check_interval_steps"]
    sched = setup.schedule
    track_codes = track and setup.bits != BF16_CONTROL_BITS
    out = Continuation([], [], RunningMean() if track else None,
                       RunningMean() if track_codes else None, [])
    model.train()
    for t in range(k):
        step = start_step + t
        lr = lr_fn(t)
        for group in opt.param_groups:
            group["lr"] = lr
        out.lrs.append(lr)

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for micro in range(sched.grad_accum):
            micro_step = step * sched.grad_accum + micro
            out.micro_steps.append(micro_step)
            x, y = TR.get_batch(train_tokens, sched, setup.seed, micro_step, device)
            _, loss = model(x, y)
            (loss / sched.grad_accum).backward()
            step_loss += loss.item() / sched.grad_accum
        if not math.isfinite(step_loss):
            raise TR.NanLoss(f"{setup.run_id}: non-finite loss at continuation step {step}")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), o_cfg["grad_clip"])
        if not torch.isfinite(grad_norm):
            raise TR.NanLoss(f"{setup.run_id}: non-finite grad norm at step {step}")
        opt.step()

        if collapse_every and (step + 1) % collapse_every == 0:
            model.check_no_collapse(t_cfg["collapse_min_nonzero_frac"])
        if out.params is not None:
            out.params.add(dict(model.named_parameters()))
        if out.codes is not None:
            out.codes.add(layer_codes(model))
        if eval_every and (t + 1) % eval_every == 0:
            out.evals.append(standard_eval(model, val_tokens, setup, device))
    return out


def _cpu_state(model: IsoMemGPT) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def process_run(row: dict, ckpt_dir: Path, out_dir: Path, train_tokens, val_tokens,
                device: torch.device, k: int) -> dict:
    """Everything M1 does for one run. Returns its m1.csv row; writes only the (c) weights."""
    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    setup = reconstruct(row)
    ckpt_path = Path(ckpt_dir) / f"{setup.run_id}.pt"
    model, opt, ck = load_checkpoint(setup, ckpt_path, device)

    last = ck["history"][-1]
    if last["step"] != ck["step"]:
        raise CheckpointProvenanceError(f"{setup.run_id}: history ends at step {last['step']}, "
                                        f"checkpoint at {ck['step']}")
    val_recorded = float(last["val"])
    out = {
        "run_id": setup.run_id, "sweep": row["sweep"], "budget_name": row["budget_name"],
        "bits": setup.bits, "seed": setup.seed, "n_layer": setup.cfg.n_layer,
        "n_embd": setup.cfg.n_embd, "val_recorded_final": val_recorded,
        "val_loss_best": float(row["val_loss_best"]), "k": k, "device": device.type,
        "git_sha": git_sha(),
    }

    # --- the setup check: the checkpoint as saved, with the standard evaluation -----------
    val_ckpt = standard_eval(model, val_tokens, setup, device)
    out["val_ckpt"] = val_ckpt
    reasons = []
    if abs(val_ckpt - val_recorded) > MISMATCH_TOL:
        reasons.append(f"val_ckpt {val_ckpt:.6f} vs checkpoint history {val_recorded:.6f}")
    # results.csv holds only the best val; it is the final one when the best step was last.
    if int(row["best_step"]) == ck["step"] and \
            abs(val_ckpt - float(row["val_loss_best"])) > MISMATCH_TOL:
        reasons.append(f"val_ckpt {val_ckpt:.6f} vs results.csv {row['val_loss_best']}")
    if reasons:
        print(f"    SETUP-MISMATCH: {'; '.join(reasons)}", flush=True)
        return _finish(out, "setup-mismatch", t0, device)

    out["val_fresh"] = evaluate_seeded(model, val_tokens, setup.schedule, N_FRESH_BATCHES,
                                       device, FRESH_EVAL_SEED)

    # --- continuation 1: constant final lr, (a) evals, (b) codes, (c) parameters ----------
    lr = final_lr(setup)
    out["lr_used"] = lr
    c1 = continue_run(model, opt, setup, train_tokens, val_tokens, device,
                      start_step=ck["step"], k=k, lr_fn=lambda t: lr,
                      eval_every=EVAL_EVERY, track=True)
    out["A_evals"] = json.dumps(c1.evals)
    out["A_mean"] = statistics.mean(c1.evals)
    out["A_sd"] = statistics.stdev(c1.evals) if len(c1.evals) > 1 else float("nan")

    final_state = _cpu_state(model)
    if c1.codes is not None:
        latents = {name: layer.weight.detach() for name, layer in model.quantized_layers()}
        qweights = mean_code_weights(c1.codes.mean(torch.float64), latents, setup.bits)
        out["val_b"] = evaluate_fixed_quantized(setup, final_state, qweights, val_tokens, device)
    averaged = c1.params.mean(torch.float32)
    out["val_c"] = evaluate_state(setup, averaged, val_tokens, device)
    # Held on the CPU, and saved only once continuation 2 has succeeded: a run that fails
    # there leaves no (c) file behind for the §14 read to pick up.
    averaged = {n: v.detach().cpu() for n, v in averaged.items()}
    del model, opt, ck, c1, final_state

    # --- continuation 2: fresh reload, lr annealed linearly to 0 --------------------------
    model, opt, ck = load_checkpoint(setup, ckpt_path, device)
    continue_run(model, opt, setup, train_tokens, val_tokens, device,
                 start_step=ck["step"], k=k, lr_fn=lambda t: anneal_lr(t, k, lr))
    out["val_d"] = standard_eval(model, val_tokens, setup, device)
    save_averaged(out_dir, setup.run_id, averaged, k)
    return _finish(out, "complete", t0, device)


def _finish(out: dict, status: str, t0: float, device: torch.device) -> dict:
    out["status"] = status
    out["wall_s"] = time.time() - t0
    cuda = device.type == "cuda"
    out["peak_mem_gib"] = torch.cuda.max_memory_allocated(device) / 2**30 if cuda else ""
    # What the process beside it competes with: the caching allocator's cudaMalloc'd total.
    out["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30 if cuda else ""
    return out


def check_software(row: dict, device: torch.device) -> None:
    """The torch (and, on CUDA, CUDA) build must be the one that trained the run.

    A different build can change kernels and so the continuation's numbers; the val_ckpt
    check would not necessarily notice. Raises, never warns (hard rule 2).
    """
    if torch.__version__ != row["torch_version"]:
        raise CheckpointProvenanceError(
            f"{row['run_id']}: torch {torch.__version__}, the run was trained with "
            f"{row['torch_version']}")
    if device.type == "cuda" and (torch.version.cuda or "") != row["cuda_version"]:
        raise CheckpointProvenanceError(
            f"{row['run_id']}: CUDA {torch.version.cuda}, the run was trained with "
            f"{row['cuda_version']}")


def save_averaged(out_dir: Path, rid: str, state: dict, k: int) -> Path:
    """(c) weights for the §14 test read. Atomic, so a kill mid-write leaves no torn file.

    torch.save writes through C++, which the audit hook does not see, so the target is
    checked against the guard explicitly.
    """
    path = Path(out_dir) / "averaged" / f"{rid}.pt"
    tmp = path.with_suffix(".tmp")
    if _ACTIVE_GUARD is not None:
        _ACTIVE_GUARD.check_write(path)
        _ACTIVE_GUARD.check_write(tmp)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {n: v.detach().cpu() for n, v in state.items()},
                "run_id": rid, "k": k, "git_sha": git_sha()}, tmp)
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------------------------


def target_rows(results_csv: Path | None = None) -> list[dict]:
    rows = [r for r in load_rows(results_csv or RESULTS_CSV)
            if r["status"] == "complete" and r["sweep"] in SWEEPS]
    if EXPECTED_RUNS is not None and len(rows) != EXPECTED_RUNS:
        raise AssertionError(f"{len(rows)} complete TinyStories runs, A7 names {EXPECTED_RUNS}")
    return rows


def run_cost(row: dict, k: int) -> float:
    """Parameters x tokens of the continuation. Small cells first."""
    return int(row["params_counted"]) * k * int(row["effective_batch_tokens"])


def order_by_cost(rows: list[dict], k: int) -> list[dict]:
    return sorted(rows, key=lambda r: (run_cost(r, k), r["run_id"]))


def recorded(out_csv: Path) -> dict[str, dict]:
    if not out_csv.exists():
        return {}
    with open(out_csv, newline="") as fh:
        return {r["run_id"]: r for r in csv.DictReader(fh)}


def append_row(out_csv: Path, row: dict) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not out_csv.exists()
    with open(out_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({f: row.get(f, "") for f in FIELDS})
        fh.flush()
        os.fsync(fh.fileno())


def select_device(allow_cpu: bool) -> torch.device | None:
    """CUDA, as every run being continued was trained on; CPU only for the tests."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu") if allow_cpu else None


def quiet_hours_block() -> str | None:
    """Why no run may start now, or None. Same deadline the sweep obeys (sweep.next_deadline),
    which already stops 10 min early; QUIET_MARGIN_S adds 10 more.

    Fails closed where the sweep does not: without WAKE_AT, next_deadline cannot tell that
    the clock is inside the quiet window and would take tomorrow's SLEEP_AT.
    """
    if quiet_hours_settings() is not None and not _quiet_hours_values().get("WAKE_AT"):
        return "quiet-hours sets SLEEP_AT but no WAKE_AT, so the window cannot be located"
    deadline = next_deadline(None, None)
    if deadline is None:
        return None
    left = deadline - time.time()
    if left < QUIET_MARGIN_S:
        return (f"{max(left, 0) / 60:.1f} min to the quiet-hours deadline"
                if left > 0 else "inside quiet hours")
    return None


def own_nonalloc_bytes(device: torch.device) -> tuple[float, str]:
    """This process's GPU memory outside the caching allocator (CUDA context, cuBLAS
    workspaces), from nvidia-smi after the first CUDA call; the assumed value otherwise."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        for line in res.stdout.splitlines():
            pid, used = (x.strip() for x in line.split(","))
            if int(pid) == os.getpid():
                return float(used) * 2**20 - torch.cuda.memory_reserved(device), "measured"
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return OWN_NONALLOC_GIB_ASSUMED * 2**30, "assumed"


@dataclass(frozen=True)
class MemoryVerdict:
    stop: str | None = None      # no run may start now: end the invocation
    skip: str | None = None      # this run does not fit now: leave it for a later tick


def memory_guard(device: torch.device, need_bytes: float, *, min_free_gib: float,
                 reserve_gib: float, arm_worst_gib: float, arm_done: bool,
                 own_nonalloc: float) -> MemoryVerdict:
    """Whether a run needing ``need_bytes`` may start now; if so, cap this process.

    The cap is the smaller of (a) what is free now plus this process's own reserved memory,
    less ``reserve_gib``; and, while the arm is not finalised, (b) the card's total less this
    process's non-allocator memory, the arm's worst-case footprint and ``reserve_gib``.
    (a) alone would let this process take memory the arm is not using at this moment (a
    256 Kbit cell, or a process still hashing tokens) and keep it for minutes, so that the
    arm's next 4 Mbit cell OOMs -- and the arm's OOM path halves its micro-batch, making that
    run numerically unlike every other. (b) does not depend on what the arm uses now.
    """
    free, total = torch.cuda.mem_get_info(device)
    if free < min_free_gib * 2**30:
        return MemoryVerdict(
            stop=f"{free / 2**30:.2f} GiB free on the GPU, below --min-free-gib {min_free_gib}")
    reserve = reserve_gib * 2**30
    allowed = free + torch.cuda.memory_reserved(device) - reserve
    if not arm_done:
        allowed = min(allowed, total - own_nonalloc - arm_worst_gib * 2**30 - reserve)
    if allowed < MIN_ALLOWED_GIB * 2**30:
        return MemoryVerdict(stop=(
            f"{max(allowed, 0) / 2**30:.2f} GiB allowed under the cap, below "
            f"{MIN_ALLOWED_GIB} ({free / 2**30:.2f} free, arm "
            f"{'finalised' if arm_done else 'live'}, arm worst {arm_worst_gib:.2f}, "
            f"reserve {reserve_gib})"))
    if need_bytes > allowed:
        return MemoryVerdict(skip=(
            f"needs ~{need_bytes / 2**30:.2f} GiB, {max(allowed, 0) / 2**30:.2f} GiB allowed "
            f"({free / 2**30:.2f} free, arm {'finalised' if arm_done else 'live'}, "
            f"arm worst {arm_worst_gib:.2f}, reserve {reserve_gib})"))
    # torch requires an indexed device here (a bare "cuda" raises ValueError).
    index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(min(1.0, allowed / total), index)
    return MemoryVerdict()


def read_oom_deferred(out_dir: Path) -> set[str]:
    path = out_dir / OOM_DEFERRED
    if not path.exists():
        return set()
    return {line.split("\t")[0] for line in path.read_text().splitlines() if line.strip()}


def add_oom_deferred(out_dir: Path, rid: str) -> None:
    with open(out_dir / OOM_DEFERRED, "a") as fh:
        fh.write(f"{rid}\t{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")


def outcome(ids: set[str], out_csv: Path) -> int:
    """0 only when every target run is recorded AND complete; 1 while any recorded row is
    not complete (setup-mismatch, nan, collapsed), so the tick never retires over one."""
    rec = recorded(out_csv)
    bad = sorted(r for r in ids & set(rec) if rec[r]["status"] != "complete")
    if bad:
        print("recorded runs that are not complete: "
              + ", ".join(f"{r} ({rec[r]['status']})" for r in bad), flush=True)
        return EXIT_ERROR
    return EXIT_OK if ids <= set(rec) else EXIT_MORE


def _release(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------


def _refuse_overlap(main_dir: Path, ckpt_dir: Path, out_dir: Path) -> str | None:
    """Why these directories are unsafe together, or None."""
    for a, b, what in ((REPO_ROOT, main_dir, "this script's copy"),
                       (out_dir, main_dir, "--out-dir"), (out_dir, ckpt_dir, "--out-dir")):
        if _under(a, b) or _under(b, a):
            other = "--main-dir" if b == main_dir else "--ckpt-dir"
            return f"{what} {a} and {other} {b} must not contain each other"
    return None


def _take_lock(out_dir: Path):
    """An exclusive lock on <out-dir>/.m1.lock, or None if another invocation holds it.

    Not a7_tick.sh's .a7.lock: flock locks belong to an open file description, so this
    process's own open of the tick's lock file would conflict with the fd it inherited.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / LOCK_NAME, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    return fh


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt-dir", type=Path, default=CORPUS.ckpt_dir)
    ap.add_argument("--main-dir", type=Path, default=None,
                    help="the main checkout, never written (default: --ckpt-dir's parent)")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "a7")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="at most this many runs now")
    ap.add_argument("--only", default=None, metavar="RUN_ID")
    ap.add_argument("--allow-cpu", action="store_true", help="tests only")
    ap.add_argument("--min-free-gib", type=float, default=1.5)
    ap.add_argument("--reserve-gib", type=float, default=1.0)
    ap.add_argument("--arm-worst-gib", type=float, default=ARM_WORST_GIB_ASSUMED,
                    help="the FineWeb-Edu arm's worst GPU footprint, measured on the host "
                         f"(default {ARM_WORST_GIB_ASSUMED:.2f}: ASSUMED, not measured)")
    ap.add_argument("--mismatches-reviewed", action="store_true",
                    help="a human has examined every setup-mismatch row in m1.csv; continue "
                         "with the other runs")
    args = ap.parse_args(argv)

    if not CORPUS.is_default:
        print(f"A7 M1 continues TinyStories runs; unset ISOMEM_DATASET (is {CORPUS.name})")
        return EXIT_ERROR
    here = Path(isomem.__file__).resolve().parent
    if here != REPO_ROOT / "isomem":
        print(f"imported isomem from {here}, not from {REPO_ROOT}")
        return EXIT_ERROR
    assert_fixed_invariants()
    if args.k < 2 or args.k % EVAL_EVERY:
        print(f"--k must be a multiple of {EVAL_EVERY} (A7: evaluate every {EVAL_EVERY} steps)")
        return EXIT_ERROR

    ckpt_dir = args.ckpt_dir.expanduser().resolve()
    main_dir = (args.main_dir.expanduser() if args.main_dir else ckpt_dir.parent).resolve()
    out_dir = args.out_dir.expanduser().resolve()
    why = _refuse_overlap(main_dir, ckpt_dir, out_dir)
    if why:
        print(why)
        return EXIT_ERROR
    out_csv = out_dir / "m1.csv"

    lock = _take_lock(out_dir)
    if lock is None:
        print(f"another invocation holds {out_dir / LOCK_NAME}; nothing done", flush=True)
        return EXIT_MORE
    try:
        return _run(args, main_dir, ckpt_dir, out_dir, out_csv)
    finally:
        lock.close()


def _run(args, main_dir: Path, ckpt_dir: Path, out_dir: Path, out_csv: Path) -> int:
    rows = target_rows()
    ids = {r["run_id"] for r in rows}
    done = recorded(out_csv)
    mismatched = sorted(r for r, v in done.items() if v["status"] == "setup-mismatch")
    if mismatched and not args.mismatches_reviewed:
        print(f"m1.csv records a setup mismatch for {mismatched}: A7 says the setup would be "
              "wrong. Nothing more runs until a human has examined it (then "
              "--mismatches-reviewed).", flush=True)
        return EXIT_ERROR
    todo = [r for r in order_by_cost(rows, args.k) if r["run_id"] not in done]
    if args.only:
        if args.only not in ids:
            print(f"{args.only} is not a complete TinyStories run")
            return EXIT_ERROR
        todo = [r for r in todo if r["run_id"] == args.only]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(rows)} target runs, {len(ids & set(done))} recorded, {len(todo)} to do now",
          flush=True)
    if not todo:
        return outcome(ids, out_csv)

    device = select_device(args.allow_cpu)
    if device is None:
        print("no CUDA device: every continued run was trained on CUDA (--allow-cpu: tests)")
        return EXIT_ERROR
    mixed = {r["device"] for r in done.values()} - {device.type}
    if mixed:
        print(f"m1.csv holds rows from {mixed}; refusing to add {device.type} rows")
        return EXIT_ERROR
    for row in todo:
        check_software(row, device)
    arm_done = (main_dir / ARM_MARKER).exists()
    print(f"device {device}   git {git_sha()}   K {args.k}   arm "
          f"{'finalised' if arm_done else 'live'}   arm worst {args.arm_worst_gib:.2f} GiB"
          f"{' (ASSUMED)' if args.arm_worst_gib == ARM_WORST_GIB_ASSUMED else ''}", flush=True)

    tokens = None
    own_nonalloc = None
    skipped = []
    oom_deferred = read_oom_deferred(out_dir)
    _activate(AccessGuard(main_dir, ckpt_dir, RESULTS_CSV))
    try:
        for i, row in enumerate(todo, 1):
            rid = row["run_id"]
            if rid in oom_deferred and not arm_done:
                skipped.append(rid)
                print(f"[{i}/{len(todo)}] {rid}: hit OOM under the cap beside the live arm; "
                      "waits for the arm", flush=True)
                continue
            why = quiet_hours_block()
            if why:
                print(f"{why}; stopping, {len(todo) - i + 1} run(s) left", flush=True)
                return EXIT_MORE
            if device.type == "cuda":
                if own_nonalloc is None:
                    torch.cuda.mem_get_info(device)          # creates the context first
                    own_nonalloc, how = own_nonalloc_bytes(device)
                    print(f"own non-allocator GPU memory {own_nonalloc / 2**30:.2f} GiB "
                          f"({how})", flush=True)
                # No per-run estimate: results.csv's peak_mem_mb is a per-process maximum
                # (every 256 Kbit row reads 987 MB), useless for sizing one run. The cap
                # below is the protection; a run that does not fit OOMs inside it.
                verdict = memory_guard(
                    device, 0.0, min_free_gib=args.min_free_gib,
                    reserve_gib=args.reserve_gib, arm_worst_gib=args.arm_worst_gib,
                    arm_done=arm_done, own_nonalloc=own_nonalloc)
                if verdict.stop:
                    print(f"{verdict.stop}; stopping, {len(todo) - i + 1} run(s) left",
                          flush=True)
                    return EXIT_MORE
                if verdict.skip:
                    skipped.append(rid)
                    print(f"[{i}/{len(todo)}] {rid}: not now, {verdict.skip}", flush=True)
                    continue
            if tokens is None:
                tokens = D.load_tokens("train"), D.load_tokens("val")
            print(f"[{i}/{len(todo)}] {rid}", flush=True)
            try:
                result = process_run(row, ckpt_dir, out_dir, *tokens, device, args.k)
            except torch.cuda.OutOfMemoryError as exc:
                # OOM inside this process's own cap: the arm was never at risk. While the
                # arm is live the run waits for it (recorded, so no tick retries it into the
                # same cap); either way this invocation stops, since the next runs in cost
                # order are mostly larger.
                _release(device)
                if not arm_done:
                    add_oom_deferred(out_dir, rid)
                print(f"    OOM under the cap, nothing recorded; stopping: "
                      f"{str(exc).splitlines()[0]}", flush=True)
                return EXIT_MORE
            except (TR.NanLoss, TernaryCollapse) as exc:
                # A failed run fails alone, as in the sweep, and is recorded as such.
                status = "nan" if isinstance(exc, TR.NanLoss) else "collapsed"
                result = {"run_id": rid, "sweep": row["sweep"],
                          "budget_name": row["budget_name"], "bits": row["bits_per_weight"],
                          "seed": row["seed"], "status": status, "k": args.k,
                          "device": device.type, "git_sha": git_sha()}
                print(f"    {status.upper()}: {exc}", flush=True)
            _release(device)
            append_row(out_csv, result)
            if result["status"] == "setup-mismatch":
                print("    setup mismatch recorded; stopping so that a human looks before "
                      "any other run is written", flush=True)
                return EXIT_ERROR
            if result["status"] == "complete":
                print(f"    val_ckpt {result['val_ckpt']:.4f}  A {result['A_mean']:.4f}  "
                      f"val_b {result.get('val_b', float('nan')):.4f}  "
                      f"val_c {result['val_c']:.4f}  val_d {result['val_d']:.4f}  "
                      f"{result['wall_s']:.0f}s", flush=True)
    finally:
        _activate(None)

    if skipped:
        print(f"not started for GPU memory: {len(skipped)} run(s)", flush=True)
    return outcome(ids, out_csv)


if __name__ == "__main__":
    raise SystemExit(main())
