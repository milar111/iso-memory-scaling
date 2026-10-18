"""A single training run: convergence protocol, guards, checkpoint/resume (SPEC §2.5, §3).

Batching is a pure function of ``(seed, micro_step)``
--------------------------------------------------
There is no dataloader state, no shuffle buffer, and no iterator to restore. Batch *k* is
the same batch on every machine, in every process, forever — so resuming from a checkpoint
reproduces an uninterrupted run exactly, and `test_resume_bitwise` tests something real
rather than testing that we remembered to save an RNG seed. Combined with zero dropout,
the only stochasticity in a run is weight initialisation.

Convergence protocol (SPEC §2.5; revised after the step-6 pilot, Amendment A3)
------------------------------------------------------------------------------
SPEC §2.5 prescribes early stopping on val loss with patience 5. **The pilot showed this
can never fire**, and that it would be harmful if it did.

A cosine schedule decaying to ``min_lr`` over ``max_steps`` makes the loss improve until the
final step by construction, so "no improvement for 5 evals" does not occur: 2 of 2 pilot
cells ran to 100% of their cap with the best val loss at 96-98% of training. Worse,
stopping early would *truncate the schedule* and leave the learning rate high, producing a
worse model than simply finishing — so early stopping does not merely fail here, it would
degrade exactly the runs it fired on.

Every run therefore completes its full schedule, and convergence is *assessed* rather than
*triggered*. Over a run's evals, with ``tail`` the final quarter and ``prev`` the quarter
before it::

    drift = mean(prev) - mean(tail)     # positive: still improving
    noise = stdev(tail)
    converged  iff  drift <= noise

"Further training moves the number by less than the noise in measuring it." Scale-free, so
it applies unchanged at every budget, and it is the same kind of quantity as the null
criterion. A run that finishes with ``drift > noise`` is marked ``undertrained`` and
excluded from the headline curve rather than quietly averaged in.

This matters more than the mechanism it replaces. At a uniform token cap, tokens-per-
parameter varies ~10x across bit-widths *within one budget* (at 828 Kbit, b=1.58 gets 572
tok/param while b=16 gets 5,634), so the low-precision configurations are systematically
the least-trained — precisely SPEC §2.5's confound. Token budgets are therefore sized from
`scripts/calibrate_convergence.py` so that even the heaviest configuration converges; every
lighter one is then converged too, and the imbalance stops mattering.
"""

from __future__ import annotations

import json
import math
import signal
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch

from isomem import data as D
from isomem.accounting import ModelConfig, assert_within_budget, accounting_row
from isomem.fixed import FIXED, learning_rate
from isomem.model import IsoMemGPT
from isomem.quant import TernaryCollapse


class NanLoss(RuntimeError):
    """Loss became non-finite. Aborts the run, never the sweep (SPEC §3)."""


class GracefulStop(Exception):
    """The run was asked to stop and has checkpointed. Resume continues it exactly.

    Raised when a wall-clock deadline passes (the machine is about to suspend for the
    night) or on SIGTERM/SIGINT. The checkpoint is written *before* this is raised, so the
    run resumes from the step it reached rather than from the last eval boundary.
    """


# Fields renamed since checkpoints were first written. A schema change must never strand a
# part-trained run: `converged` became `tail_drift_below_noise` when Amendment A3 demoted it
# from a verdict to a diagnostic.
_RENAMED_RESULT_FIELDS = {"converged": "tail_drift_below_noise"}


def _coerce_result(stored: dict) -> dict:
    """Adapt a checkpoint's stored result row to the current RunResult schema.

    Renamed fields are mapped and unknown ones dropped, so a checkpoint written by an older
    build still resumes instead of raising. Losing hours of training to a field rename is
    not an acceptable failure mode.
    """
    import dataclasses

    valid = {f.name for f in dataclasses.fields(RunResult)}
    out = {}
    for key, value in stored.items():
        key = _RENAMED_RESULT_FIELDS.get(key, key)
        if key in valid:
            out[key] = value
    return out


def tail_convergence(history: list[dict], quarter: int | None = None) -> tuple[float, float, bool]:
    """(drift, noise, converged) from a run's eval history. See the module docstring.

    Returns ``(nan, nan, False)`` when there are too few evals to judge — a run that never
    produced enough evaluations cannot be declared converged.
    """
    vals = [h["val"] for h in history]
    if len(vals) < 4:
        return float("nan"), float("nan"), False
    q = quarter or max(2, len(vals) // 4)
    tail, prev = vals[-q:], vals[-2 * q : -q]
    if not prev:
        return float("nan"), float("nan"), False
    drift = statistics.mean(prev) - statistics.mean(tail)
    noise = statistics.pstdev(tail)
    return drift, noise, bool(drift <= noise)


# --------------------------------------------------------------------------------------
# Device and batching
# --------------------------------------------------------------------------------------


def resolve_device(prefer: str | None = None) -> torch.device:
    """cuda -> mps -> cpu, resolved at runtime and logged per run. Never hardcoded."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class BatchSchedule:
    """Micro-batch and accumulation, chosen so the effective batch never varies.

    SPEC §3 (OOM row): on OOM, halve the micro-batch and double accumulation. The effective
    batch must stay constant across every run, or the comparison is meaningless.
    """

    micro_batch: int
    grad_accum: int
    block_size: int

    @property
    def effective_batch_tokens(self) -> int:
        return self.micro_batch * self.grad_accum * self.block_size

    def halved(self) -> "BatchSchedule":
        if self.micro_batch % 2 != 0:
            raise ValueError(f"cannot halve an odd micro_batch ({self.micro_batch})")
        return BatchSchedule(self.micro_batch // 2, self.grad_accum * 2, self.block_size)


def make_batch_schedule(
    effective_batch_tokens: int, micro_batch: int, block_size: int
) -> BatchSchedule:
    per_micro = micro_batch * block_size
    if effective_batch_tokens % per_micro != 0:
        raise ValueError(
            f"effective_batch_tokens={effective_batch_tokens} is not divisible by "
            f"micro_batch*block_size={per_micro}; the effective batch would differ per run"
        )
    return BatchSchedule(micro_batch, effective_batch_tokens // per_micro, block_size)


def _batch_seed(seed: int, micro_step: int) -> int:
    """Deterministic, well-separated per-(seed, micro_step) stream."""
    return (seed * 1_000_003 + micro_step * 2_654_435_761) % (2**63 - 1)


def get_batch(
    tokens: np.ndarray, schedule: BatchSchedule, seed: int, micro_step: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch `micro_step` of the run with seed `seed`. A pure function of its arguments."""
    g = torch.Generator().manual_seed(_batch_seed(seed, micro_step))
    hi = len(tokens) - schedule.block_size - 1
    ix = torch.randint(hi, (schedule.micro_batch,), generator=g).tolist()
    bs = schedule.block_size
    x = np.stack([np.asarray(tokens[i : i + bs], dtype=np.int64) for i in ix])
    y = np.stack([np.asarray(tokens[i + 1 : i + 1 + bs], dtype=np.int64) for i in ix])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def lr_at(step: int, max_steps: int, base_lr: float) -> float:
    """Warmup then cosine decay. One schedule for every run (SPEC §2.8)."""
    cfg = FIXED["lr"]
    warmup = max(1, int(cfg["warmup_frac"] * max_steps))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, progress)
    min_lr = base_lr * cfg["min_lr_frac"]
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------

EVAL_SEED = 987_654_321  # fixed, so every eval sees the same val batches at every step


@torch.no_grad()
def evaluate(
    model: IsoMemGPT,
    tokens: np.ndarray,
    schedule: BatchSchedule,
    n_batches: int,
    device: torch.device,
) -> float:
    """Mean val loss over a fixed set of batches. Deterministic across steps and runs."""
    was_training = model.training
    model.eval()
    total = 0.0
    for i in range(n_batches):
        x, y = get_batch(tokens, schedule, EVAL_SEED, i, device)
        _, loss = model(x, y)
        total += loss.item()
    model.train(was_training)
    return total / n_batches


# --------------------------------------------------------------------------------------
# Run state
# --------------------------------------------------------------------------------------


@dataclass
class RunResult:
    run_id: str
    status: str = "running"
    seed: int = 0
    steps: int = 0
    tokens_seen: int = 0
    stopped_early: bool = False
    undertrained: bool = False
    train_loss: float = float("nan")
    val_drift: float = float("nan")
    val_noise: float = float("nan")
    # Diagnostic ONLY. Named for what it measures rather than "converged", because under a
    # cosine schedule tail drift tracks schedule shape, not convergence (Amendment A3).
    tail_drift_below_noise: bool = False
    val_loss_best: float = float("inf")
    val_ppl_best: float = float("inf")
    best_step: int = 0
    wall_clock_s: float = 0.0
    peak_mem_mb: float = 0.0
    device: str = ""
    lr: float = 0.0
    history: list[dict] = field(default_factory=list)

    def as_row(self) -> dict:
        row = asdict(self)
        row.pop("history")
        return row


def _peak_mem_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 1e6
    return 0.0


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------


def train_one(
    cfg: ModelConfig,
    bits: float,
    seed: int,
    budget_bits: int,
    max_tokens: int,
    run_id: str = "adhoc",
    ckpt_dir: Path | None = None,
    device: torch.device | None = None,
    train_tokens: np.ndarray | None = None,
    val_tokens: np.ndarray | None = None,
    micro_batch: int | None = None,
    eval_interval: int | None = None,
    eval_batches: int | None = None,
    patience: int | None = None,
    max_steps_override: int | None = None,
    log_every: int = 0,
    compile_mode: str | None = None,
    deadline: float | None = None,
    checkpoint_every_s: float = 600.0,
    lr_multiplier: float = 1.0,
) -> RunResult:
    """Train one configuration to convergence. Resumable and idempotent.

    ``lr_multiplier`` scales the learning-rate rule's value, and exists for one purpose
    only: the learning-rate sensitivity arm of PREREGISTRATION.md §16 (Amendment A7, M2,
    ``scripts/a7_followup.py``), whose runs carry their own run ids and results file and
    never enter the primary analysis. Nothing else passes it (tested), and the default 1.0
    leaves training bitwise identical to this function before the parameter existed (tested
    against that commit's train.py: parameters, optimizer state, history, result row).
    """
    if not (math.isfinite(lr_multiplier) and lr_multiplier > 0):
        raise ValueError(f"lr_multiplier must be a positive finite number, got {lr_multiplier}")
    t_cfg = FIXED["training"]
    o_cfg = FIXED["optimizer"]
    micro_batch = micro_batch or t_cfg["micro_batch"]
    eval_interval = eval_interval or t_cfg["eval_interval_steps"]
    eval_batches = eval_batches or t_cfg["eval_batches"]
    # Retained for signature compatibility and ignored: convergence is assessed at the end
    # of the run, not triggered during it (Amendment A3).
    del patience
    collapse_every = t_cfg["collapse_check_interval_steps"]

    device = device or resolve_device()
    # Crash before training if the configuration misses its budget (SPEC §2.1).
    assert_within_budget(cfg, bits, budget_bits)

    seed_everything(seed)
    model = IsoMemGPT(cfg, bits, rope_theta=FIXED["model"]["rope_theta"]).to(device)
    model.assert_matches_accounting()
    # torch.compile is OPT-IN and off by default. At these model sizes the bottleneck is
    # kernel-launch overhead, which CUDA graphs ("reduce-overhead") can collapse -- but only
    # adopt it if it is ALSO bitwise deterministic, since resume and reproducibility depend
    # on that. Guards below call the uncompiled module so they see the real parameters.
    train_model = torch.compile(model, mode=compile_mode) if compile_mode else model

    schedule = make_batch_schedule(t_cfg["effective_batch_tokens"], micro_batch, cfg.block_size)
    tokens_per_step = schedule.effective_batch_tokens
    max_steps = max_steps_override or max(1, max_tokens // tokens_per_step)

    base_lr = learning_rate(cfg.n_embd) * lr_multiplier
    opt = torch.optim.AdamW(
        model.param_groups(o_cfg["weight_decay"]),
        lr=base_lr,
        betas=(o_cfg["beta1"], o_cfg["beta2"]),
    )

    if train_tokens is None:
        train_tokens = D.load_tokens("train")
    if val_tokens is None:
        val_tokens = D.load_tokens("val")

    result = RunResult(run_id=run_id, seed=seed, device=str(device), lr=base_lr)
    start_step = 0

    ckpt_path = (ckpt_dir / f"{run_id}.pt") if ckpt_dir else None
    if ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_step = state["step"]
        result = RunResult(**_coerce_result(state["result"]),
                           history=state.get("history", []))
        result.status = "running"

    t0 = time.time()
    recent: list[dict] = []  # last 10 steps, dumped if the loss goes non-finite
    model.train()

    # A SIGTERM (systemd stopping us before the machine suspends) must not lose a run.
    stop_requested = {"flag": False}

    def _request_stop(signum, _frame):
        stop_requested["flag"] = True

    previous_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_handlers[sig] = signal.signal(sig, _request_stop)
        except ValueError:
            pass  # not on the main thread; deadline checking still applies

    last_ckpt = time.time()

    def save_checkpoint(step_done: int) -> None:
        if not ckpt_path:
            return
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step_done,
                "result": result.as_row(),
                "history": result.history,
            },
            tmp,
        )
        tmp.replace(ckpt_path)  # atomic: a suspend mid-write must not corrupt the file

    for step in range(start_step, max_steps):
        lr = lr_at(step, max_steps, base_lr)
        for group in opt.param_groups:
            group["lr"] = lr

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for micro in range(schedule.grad_accum):
            micro_step = step * schedule.grad_accum + micro
            x, y = get_batch(train_tokens, schedule, seed, micro_step, device)
            _, loss = train_model(x, y)
            # Scale so the gradient equals that of one effective batch, whatever the split.
            (loss / schedule.grad_accum).backward()
            step_loss += loss.item() / schedule.grad_accum

        recent.append({"step": step, "loss": step_loss, "lr": lr})
        recent = recent[-10:]
        if not math.isfinite(step_loss):
            result.status = "nan"
            raise NanLoss(
                f"non-finite loss at step {step} (run {run_id}). Last 10 steps:\n"
                + "\n".join(json.dumps(r) for r in recent)
            )

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), o_cfg["grad_clip"])
        if not torch.isfinite(grad_norm):
            result.status = "nan"
            raise NanLoss(
                f"non-finite gradient norm at step {step} (run {run_id}). Last 10 steps:\n"
                + "\n".join(json.dumps(r) for r in recent)
            )
        opt.step()

        result.steps = step + 1
        result.tokens_seen = result.steps * tokens_per_step
        result.train_loss = step_loss

        if collapse_every and (step + 1) % collapse_every == 0:
            model.check_no_collapse(t_cfg["collapse_min_nonzero_frac"])

        # Checkpoint on a wall-clock cadence, not only at eval boundaries, so an
        # interrupted run loses minutes rather than a fraction of the whole run.
        if ckpt_path and time.time() - last_ckpt >= checkpoint_every_s:
            save_checkpoint(step + 1)
            last_ckpt = time.time()

        # Stop cleanly when the night is about to start, or when asked to.
        out_of_time = deadline is not None and time.time() >= deadline
        if out_of_time or stop_requested["flag"]:
            save_checkpoint(step + 1)
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            result.status = "interrupted"
            raise GracefulStop(
                f"{run_id}: stopped at step {step + 1}/{max_steps} "
                f"({'deadline' if out_of_time else 'signal'}); checkpoint written"
            )

        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step + 1}/{max_steps} loss {step_loss:.4f} lr {lr:.2e}", flush=True)

        if (step + 1) % eval_interval == 0 or step + 1 == max_steps:
            val_loss = evaluate(model, val_tokens, schedule, eval_batches, device)
            result.history.append({"step": step + 1, "train": step_loss, "val": val_loss})
            if val_loss < result.val_loss_best:
                result.val_loss_best = val_loss
                result.val_ppl_best = math.exp(val_loss)
                result.best_step = step + 1

            save_checkpoint(step + 1)
            last_ckpt = time.time()

            # Deliberately no early-stop break: a cosine schedule must run to completion,
            # or the model is left at a high learning rate and is worse than if it had
            # simply finished (Amendment A3).

    # Under Amendment A3 the token budget is what guarantees convergence: it is solved per
    # cell so every cell lands at the same residual. So "undertrained" means exactly one
    # thing -- the run did not receive the budget it was given -- and NOT the tail-drift
    # test, which A3 disproved for cosine schedules and which is kept only as a diagnostic.
    result.val_drift, result.val_noise, result.tail_drift_below_noise = tail_convergence(
        result.history
    )
    result.undertrained = result.steps < max_steps
    result.status = "complete"
    for sig, handler in previous_handlers.items():
        signal.signal(sig, handler)
    result.wall_clock_s = time.time() - t0
    result.peak_mem_mb = _peak_mem_mb(device)
    return result


def train_row(cfg: ModelConfig, bits: float, budget_bits: int, result: RunResult) -> dict:
    """Merge accounting and run columns into one `results.csv` row (SPEC §5)."""
    return {**accounting_row(cfg, bits, budget_bits), **result.as_row()}
