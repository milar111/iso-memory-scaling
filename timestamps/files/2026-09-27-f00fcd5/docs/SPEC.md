# Implementation Prompt — Iso-Memory Scaling Laws for Quantized Transformers (Phase 1)

Hand this to a coding agent, or work through it yourself. Everything here is Phase 1: **software only, no FPGA.** Phase 2 handoff requirements are in §9 so Phase 1 emits what the hardware will need.

---

## 0. Brief for the implementer

You are building a **scientific experiment harness**, not a product and not a fast trainer. The deliverable is a defensible curve with error bars, plus the code that proves it wasn't an artifact.

**Research question.** Given a fixed budget of *B bits* of weight storage, what bits-per-weight `b` produces the best language model? Vary `b`, adjust the architecture so total weight bits stay ≈ `B`, train each to convergence, measure quality.

**Primary hypothesis under test.** The quantization literature reports results at *matched parameter count* ("our 1.58-bit model matches fp16 at the same parameter count"). Deployment on SRAM-based hardware constrains *bits*, not parameters. If the memory-matched ranking differs from the parameter-matched ranking, a subfield is reporting on the wrong axis. That is the finding.

**A null result is a publishable result.** If the curve is flat within seed noise, that is itself informative ("precision allocation doesn't matter below X"). Do not tune the experiment until it produces a slope. See §2.9.

**Success criteria.**

1. Every methodological rule in §2 is enforced in code, not in a comment.
2. Every test in §6 passes.
3. `results.csv` contains ≥3 seeds per configuration with actual (not target) bit counts logged.
4. A preregistration file exists, git-committed, timestamped before the main sweep ran.

---

## 1. Experimental design

### 1.1 Axes

| Axis | Values | Notes |
|---|---|---|
| Bits per weight `b` | 1.58, 2, 3, 4, 6, 8, 16 | 1.58 = ternary {-1,0,+1}. 16 = bf16 control. |
| Budget `B` | 256 Kbit, 828 Kbit, 4 Mbit | 828 Kbit = Tang Nano 20K BSRAM. Others check whether the optimum shifts with scale. |
| Seed | 3 minimum, 5 preferred | Non-negotiable. See §2.9. |

Full grid: 7 × 3 × 3 = **63 runs**. Budget ~1 GPU-hour each at the two smaller budgets, ~3h at 4 Mbit. Total roughly 100–150 GPU-hours. On a rented single A10/4090 that is a long weekend and $50–150.

### 1.2 Held fixed across every single run

Changing any of these invalidates cross-run comparison. Assert them at startup.

- **Tokenizer.** Train one BPE tokenizer once, vocab 512, on the training split only. Version it, hash it, commit the hash. Perplexity across different vocabularies is not comparable — this is a correctness issue, not a stylistic one.
- **Dataset and splits.** TinyStories. Fixed train/val/test split by seeded permutation. Hash each split and assert the hash at load time.
- **`block_size`** = 256. Perplexity is not comparable across context lengths.
- **Positional encoding** = RoPE. Learned positional embeddings would consume budget and vary with width, adding a confound. RoPE is free.
- **Biases** = none anywhere. **Norm** = RMSNorm, gains kept in bf16 and counted (they're negligible but count them anyway).
- **Weight tying** = embedding and output head tied. Counted once.
- **Optimizer** = AdamW, β=(0.9, 0.95), wd=0.1, grad clip 1.0.
- **LR rule.** One deterministic function of width: `lr = base_lr * (64 / n_embd)**0.5`, `base_lr = 3e-3`. Applied identically everywhere. See §2.8 on why you must not tune per config.

### 1.3 Architecture search under the bit constraint

Free knobs: `n_layer`, `n_embd`, `mlp_ratio ∈ {2.0, 2.5, 3.0, 3.5, 4.0}`. `n_head` derived as `n_embd // 32`.

Counted parameters, per the accounting in §2.1:

```
params = n_layer * (4 * n_embd**2 + 2 * mlp_ratio * n_embd**2) + vocab * n_embd
```

**You will not hit the target exactly.** `n_layer` and `n_embd` are coarse integers, which is why `mlp_ratio` is a third knob. Worked example at B = 828 Kbit, vocab 512:

- `b=2` → target 414,000 params. `n_layer=3, n_embd=96, mlp_ratio=4.0` → 380,928 params → 761,856 bits = **92% of budget.**

92% is outside a ±2% tolerance. Handle it like this:

- Search all valid `(n_layer, n_embd, mlp_ratio)` triples, pick the one minimising `|actual_bits - B|`.
- Accept ±5%. Reject and raise above that.
- **Plot against actual bits, never target bits.** Log both columns.
- Emit a table of chosen configs for human review *before* the sweep runs.

Constraints to assert during search: `n_embd % 32 == 0`, `n_embd >= 32`, `n_layer >= 2`, `n_head >= 1`.

### 1.4 Baselines

Include both, or you cannot detect a floor effect:

- **Bigram count model** on the same tokenizer. Any config not beating this is degenerate; flag it in the results.
- **bf16 control** at each budget (the `b=16` row).

---

## 2. Methodological rules — enforce in code

These are the rules that decide whether the result is real. Each one has bitten someone.

### 2.1 Bit accounting must be audited, not assumed

Write `count_weight_bits(model, bits_per_weight) -> int`.

**Counted:** all linear layer weights (qkv projection, attention output projection, MLP up, MLP down), the tied embedding/head matrix, RMSNorm gains.

**Excluded but logged in a separate column:** KV cache bits, peak activation bits, optimizer state.

KV cache exclusion is a deliberate scope decision and a stated limitation. At fixed budget, wider models carry larger KV caches, so a full hardware accounting would penalise them. Log `kv_bits_at_full_context` so a reviewer can see the magnitude and so Phase 2 can plan around it.

Assert `abs(count_weight_bits(...) - B) / B <= 0.05` before training starts. Crash otherwise. Do not warn — crash.

### 2.2 The quantizer must actually quantize

The most common silent failure in QAT code: the forward pass uses unquantized weights, gradients look healthy, loss curves look great, and you have measured nothing.

Required properties, each with a test in §6:

- After quantization, a `b`-bit tensor contains **at most 2^b unique values** (at most 3 for ternary).
- Quantization is **idempotent**: `q(q(x)) == q(x)`.
- The STE passes a **nonzero gradient** to the latent fp32 weight.
- Latent weights are fp32; quantization happens in the forward pass each step.

Use **absmean** scaling for ternary (per BitNet), not absmax. Absmax on a tensor with one outlier drives every other weight to zero.

### 2.3 QAT from step 0, never post-training quantization

At 300K parameters there is no redundancy to exploit. PTQ below 4 bits will destroy the model and you will conclude low precision is bad when you have only measured that PTQ is bad. Quantize in the forward pass from the first step.

### 2.4 Weights-only quantization

Activations stay bf16. Rationale: the question is about SRAM for weights. State it as a limitation; an activation ablation is future work, not scope.

### 2.5 Training budget is the biggest confound

If every config trains for the same number of steps, the higher-parameter-count (lower-precision) configs are undertrained and you have measured optimization speed, not capacity.

**Protocol (primary): train to convergence.** Generous token budget, early stop on val loss with patience 5 evals, report best val loss achieved. Log `tokens_seen` and `stopped_early` per run. If a run hits the token cap without triggering early stop, it is **undertrained — mark it and exclude it from the headline curve.**

**Protocol (secondary, optional): compute-matched.** Equal FLOPs across configs. Report separately. Never mix the two in one plot.

### 2.6 Never touch the test set until the end

Val for early stopping and model selection. Test loaded exactly once, at write-up. A test asserts the test split is not read during training (§6).

### 2.7 Split integrity

Hash train/val/test at creation. Assert hashes match on every load. A silently reshuffled val split makes every number incomparable and is nearly invisible.

### 2.8 Do not tune hyperparameters per configuration

Per-config tuning is how you manufacture whichever conclusion you want, usually without noticing. Use the single LR rule in §1.2 for all runs. If it must be tuned, tune it on **one** held-out configuration and apply the resulting rule unchanged everywhere, and say so in the write-up.

### 2.9 Preregister, then define your null

Before running the main sweep, write `PREREGISTRATION.md` and git-commit it:

- Predicted curve shape and the reasoning.
- **Null criterion, stated numerically:** "If the seed standard deviation at any budget exceeds the val-loss gap between adjacent bit-widths, the result is null at that budget and will be reported as null."
- Which analyses are planned versus exploratory.

Then report what happened, including if it was boring.

### 2.10 The embedding-bound regime

At small `B` and low `b`, the embedding table can dominate the budget. This is not a bug — it is a finding, and it applies directly to real SRAM accelerators. Detect when `embedding_bits / total_bits > 0.5`, log the flag, and report those points as a separate regime rather than dropping them.

---

## 3. Edge cases: detection and response

| Failure | Detection | Response |
|---|---|---|
| Ternary layer collapses to all zeros | Assert nonzero fraction > 5% per layer, checked every 100 steps | Absmean scaling; lower LR; crash with the layer name |
| Loss → NaN/Inf | `torch.isfinite(loss)` every step | Grad clip 1.0; bf16 not fp16; log last 10 steps and abort the run, not the sweep |
| STE gradient dead | Unit test (§6) | Fix before any training |
| Quantizer silently a no-op | Cardinality test (§6) | Fix before any training |
| Iso-bit target unreachable | Search returns best `> 5%` error | Widen `mlp_ratio` set; if still failing, drop that grid cell and document why |
| `n_embd % n_head != 0` | Assert in config search | Restrict search space |
| Floor effect (config too small to learn) | Compare to bigram baseline | Flag row; do not silently include |
| Overfitting tiny data | Val loss rising while train falls | Early stopping handles it; log the gap |
| Undertrained run | Hit token cap without early stop | Mark `undertrained=True`, exclude from headline curve |
| Sweep dies at run 41 of 63 | Deterministic run IDs; append-only results with a `status` column | Resume skips `status='complete'`; idempotent |
| Nondeterminism | Determinism test (§6) | Seed python/numpy/torch/cuda; log library versions; accept residual cudnn nondeterminism and report it |
| Silent dtype upcast | Cardinality test | — |
| Cherry-picked checkpoint | Val for selection, test for reporting | Enforced by §2.6 test |
| OOM at B=4 Mbit | Catch, halve batch, raise grad accumulation to keep effective batch fixed | Effective batch must stay constant across runs |

---

## 4. Repository layout

```
isomem/
  PREREGISTRATION.md          # committed before the sweep
  README.md                   # headline numbers, how to reproduce
  pyproject.toml
  config/
    grid.yaml                 # axes, budgets, seeds
    fixed.yaml               # everything from §1.2
  isomem/
    tokenizer.py              # train once, hash, freeze
    data.py                   # splits + hash assertions
    quant.py                  # STE fake-quant, absmean scaling
    accounting.py             # count_weight_bits, kv_bits
    search.py                 # architecture search under bit constraint
    model.py                  # nanoGPT-style, RoPE, RMSNorm, no biases
    train.py                  # single run, checkpoint/resume
    sweep.py                  # orchestration, idempotent, resumable
    eval.py                   # ppl, bigram baseline, story-quality harness
    export.py                 # Phase 2 artifacts (§9)
  scripts/
    make_tokenizer.py
    dry_run.py                # validate whole grid, train nothing
    plot.py
  tests/
  results/
    results.csv
    runs.sqlite
```

### Key interfaces

```python
# accounting.py
def count_weight_bits(cfg: ModelConfig, bits: float) -> int: ...
def kv_bits_at_full_context(cfg: ModelConfig, dtype_bits: int = 16) -> int: ...

# search.py
def find_config(budget_bits: int, bits: float, vocab: int,
                tol: float = 0.05) -> tuple[ModelConfig, int]:
    """Returns config and ACTUAL bits. Raises if no config within tol."""

# quant.py
class FakeQuant(torch.autograd.Function):
    """STE. Forward quantizes to <=2**b levels; backward passes straight through."""

# sweep.py
def run_id(budget: int, bits: float, seed: int) -> str:
    """Deterministic. Same inputs -> same ID, always."""
```

### `dry_run.py` — build this first

Validates the entire grid without training anything: runs the architecture search for all 63 cells, prints target vs actual bits and the percentage error, flags embedding-bound cells, estimates total GPU-hours and cost, and exits nonzero if any cell fails tolerance. **You should be able to review and sign off on the whole experiment before spending a single GPU-hour.**

---

## 5. Results schema

`results.csv`, append-only, one row per run:

```
run_id, status, budget_bits_target, budget_bits_actual, bits_per_weight,
n_layer, n_embd, n_head, mlp_ratio, vocab, params_counted,
embedding_bits, embedding_frac, kv_bits_full_ctx,
seed, tokens_seen, steps, stopped_early, undertrained,
train_loss, val_loss_best, val_ppl_best, test_loss, test_ppl,
bigram_baseline_ppl, beats_baseline,
wall_clock_s, peak_mem_mb, torch_version, cuda_version, git_sha
```

`git_sha` per row is not optional. If you change the code mid-sweep you need to know which rows came from which version.

---

## 6. Test suite

All must pass before the main sweep. Tests marked **[blocking]** must pass before *any* training.

**Accounting**

- `test_bit_accounting_exact` **[blocking]** — two hand-computed configs, exact match, no tolerance. Put the arithmetic in the docstring.
- `test_kv_accounting` — scales linearly with context and width as expected.
- `test_embedding_frac` — matches manual computation.

**Quantizer**

- `test_cardinality` **[blocking]** — for b in {1.58, 2, 3, 4, 8}: `len(unique(q(x))) <= 2**ceil(b)`; exactly `<= 3` for ternary.
- `test_idempotent` **[blocking]** — `q(q(x)) == q(x)`.
- `test_ste_gradient_flows` **[blocking]** — `x.grad` is not None and not all-zero after backward.
- `test_absmean_outlier_robustness` — tensor with one 100× outlier still yields >5% nonzero ternary weights.
- `test_quantizer_is_not_identity` **[blocking]** — `q(x) != x` for random continuous `x`.

**Config search**

- `test_search_within_tolerance` — every grid cell returns within 5%, or raises.
- `test_search_constraints` — `n_embd % 32 == 0`, `n_head >= 1`, `n_layer >= 2` for all emitted configs.
- `test_search_deterministic` — same inputs, same config, every time.

**Data**

- `test_split_hashes_stable` — reconstructing splits reproduces committed hashes.
- `test_splits_disjoint` — zero document overlap between train/val/test.
- `test_no_test_leakage` **[blocking]** — monkeypatch the test loader to raise; run 50 training steps; assert it was never called.
- `test_tokenizer_hash` — matches the committed hash.

**Training**

- `test_smoke_20_steps` — loss finite, and lower at step 20 than step 1.
- `test_nan_guard` — inject an inf gradient, assert the run aborts cleanly with a useful message.
- `test_zero_collapse_guard` — force absmax scaling with an outlier, assert the collapse assertion fires.
- `test_resume_bitwise` — checkpoint at step 50, kill, resume; loss at step 60 matches an uninterrupted run to 1e-5.
- `test_determinism` — same seed twice → identical loss to 1e-4.
- `test_effective_batch_invariant` — halving batch size and doubling accumulation leaves effective batch unchanged.

**Sweep**

- `test_run_id_deterministic`
- `test_resume_skips_complete` — pre-seed results with 5 complete rows, assert they are not rerun.
- `test_dry_run_no_training` — monkeypatch the training entry point to raise; `dry_run.py` must still exit 0.

**Evaluation**

- `test_bigram_baseline_sane` — perplexity between 1 and vocab size.
- `test_ppl_matches_loss` — `ppl == exp(loss)` to 1e-6.

---

## 7. Milestones with gates

Do not pass a gate until its condition holds.

| Week | Work | Gate |
|---|---|---|
| 1 | Tokenizer, data, splits, hashing, accounting | `test_bit_accounting_exact` passes; hashes committed |
| 2 | Quantizer + all **[blocking]** tests | Every blocking test green. **No training before this.** |
| 3 | Model, single-run training loop | 20-step smoke test; loss decreases at b=2 and b=16 |
| 4 | Architecture search + `dry_run.py` | Full 63-cell grid validated; config table reviewed by a human |
| 5 | Preregistration | `PREREGISTRATION.md` committed with the numeric null criterion |
| 6–7 | Convergence protocol tuning on **one** held-out cell | Early stopping reliably triggers before the token cap |
| 8–10 | Main sweep, 63 runs | ≥3 seeds per cell; no cell marked undertrained |
| 11 | Analysis, plots, story-quality eval | Curves with error bars; null criterion applied honestly |
| 12–14 | Write-up, arXiv preprint | — |

Weeks 2 and 5 are the ones people skip. Week 2 protects you from measuring nothing; week 5 protects you from fooling yourself.

---

## 8. Analysis and plots

- **Headline:** val loss vs *actual* bits-per-weight, one line per budget, shaded ±1 seed std. Annotate the best point per budget.
- **The money plot:** the same data plotted twice side by side — matched-parameter-count on the left, matched-bits on the right. If the argmin moves between panels, that is the paper.
- Val loss vs actual bit count, colored by `b`, to show the iso-budget lines are genuinely iso.
- Embedding-bound points marked distinctly on every plot.
- Report the seed std explicitly next to every claimed gap. If they overlap, say so in the abstract.

---

## 9. Phase 2 handoff — build these exports now

Phase 1 must emit what the FPGA work will need, or you will retrain later. `export.py` writes:

1. **Winning config JSON** — full architecture and quantization parameters for the best point at B = 828 Kbit.
2. **Weights as packed bit arrays** — exactly the layout the RTL will read from BSRAM, plus per-tensor scale factors. Little-endian, documented, with a reference unpacker.
3. **Golden test vectors** — 100 token sequences with their exact logits from the quantized model, in fixed point. This is how you verify the Verilog: the RTL must reproduce these bit-for-bit. Without them you will be debugging hardware against an unknown reference, which is how FPGA projects die.
4. **Fixed-point reference implementation** — pure-integer NumPy forward pass, no floats anywhere, matching the golden vectors. This is the specification the RTL implements. Write it in Phase 1 while the model is still fresh.
5. **Operator profile** — MAC count, memory traffic, and serial-dependency depth per operator, so Phase 2 knows whether softmax or the matmuls dominate before committing to an architecture.

Item 4 is the single most valuable Phase 2 artifact. An integer-only Python model that produces the golden vectors turns hardware bring-up from guesswork into a diff.

---

## 10. Out of scope — do not build these

Scope creep is the main risk after week 8.

- Activation quantization (stated limitation, not scope)
- Sparsity or pruning
- Anything above 4 Mbit
- Any dataset other than TinyStories
- Custom CUDA kernels
- A training framework, a config system abstraction, or a plugin architecture
- Any FPGA work whatsoever

If you find yourself building infrastructure rather than running the experiment, stop and run `dry_run.py` instead.
