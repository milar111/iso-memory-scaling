# iso-memory-scaling

Code, configuration and results for *Iso-Memory Scaling Laws for Quantized Transformers*
(Daniel Yordanov).

At a fixed budget of B bits of weight storage, which bits per weight b gives the best small
language model? Decoder-only transformers are trained with weight quantization from step 0,
for b in {1.58, 2, 3, 4, 6, 8, 16} at B in {256 Kbit, 828 Kbit, 4 Mbit}. The architecture is
resized so that the stored weight bits match B, and the models are compared on validation loss.

## Contents

| Path | What it holds |
|---|---|
| `isomem/` | The library: bit accounting, architecture search, quantizer, model, training, evaluation, sweep, analysis. |
| `config/` | Fixed hyperparameters, the budget and bit-width grid, and the second-corpus configuration. The hash of `fixed.yaml` is logged with every run. |
| `scripts/` | Data preparation, the convergence pilot and calibration, the follow-up runs (A7, A8), the analysis, the figures, the single test-split read, and the diagnostic checks reported in the paper. |
| `results/` | `results.csv` (one row per run), `fineweb_edu/` and `a7/` (the second corpus and the follow-up runs), `analysis.json`, `histories.json`, `test_read.json`, and the output files of the diagnostic checks. |
| `data/` | The tokenizer and the split manifests with their hashes. The text itself is not included; `scripts/prepare_data.py` downloads it. |
| `technical-report.pdf` | The long version of the paper: full derivations, every exploratory analysis, the per-amendment trail and the limitations. |
| `PREREGISTRATION.md` | The preregistration with its dated amendments. |
| `timestamps/` | OpenTimestamps manifests and proofs, and the file versions their hashes cover. `files/2026-09-27-f00fcd5/docs/SPEC.md` is the specification the code comments refer to as SPEC. |

Model checkpoints are not included, so the scripts that read checkpoints (`build_histories.py`,
`bf16_cast_check.py`, `clip_share.py`, `a7_average.py`, `test_read.py` and others) cannot be
rerun from this repository alone; their outputs are in `results/`.

## Running the experiment

Python 3.12 and a CUDA GPU are expected for training; the device is resolved at run time and
logged with every run. The package versions of the training environment are in
`results/environment/pip-freeze-training-host.txt`.

```
python3.12 -m venv .venv && . .venv/bin/activate && pip install -e .
python scripts/prepare_data.py --force   # splits, tokenizer, token streams; rewrites the hashes
python scripts/make_baseline.py          # bigram floor
python scripts/dry_run.py                # search and validate every cell; trains nothing
python -m isomem.sweep                   # primary and secondary sweeps; resumable
python scripts/analyze.py --json results/analysis.json --histories results/histories.json
python scripts/plot.py --histories results/histories.json
```

`ISOMEM_DATASET` must be unset for the TinyStories sweep; `ISOMEM_DATASET=fineweb_edu` selects
the second corpus.

## Provenance

The `git_sha` column of `results/results.csv` names commits of a development history that is
not released, so those identifiers do not resolve here. What can be checked here is in
`timestamps/`: each manifest lists SHA-256 hashes of files, `files/` holds those file versions,
and the `.ots` file is the OpenTimestamps proof of the manifest.

## Licence

MIT; see `LICENSE`.
