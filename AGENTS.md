# AGENTS.md

Agent instructions for ContextSplice contributors.

## First Read

- Project overview and environment notes: [README.md](README.md)
- Main runnable task definitions: [Makefile](Makefile)
- Hydra defaults and runtime knobs: [config.yaml](config.yaml)

## Quick Start Commands

- List available workflows:
  - `make help`
- Install dependencies (conda env `contextsplice` by default):
  - `make install`
- Typical training entrypoint (through Make target):
  - `make train_pangolinomni3_pangolinsolo123`
- Typical evaluation entrypoint:
  - `make test_pangolinomni3_pangolinsolo123`

Prefer existing Make targets instead of inventing new command lines when reproducing experiments.

## Runtime Architecture

- Main entrypoint: [run.py](run.py)
  - Uses Hydra (`config.yaml`) and dispatches by `stage` (`train`/`eval`/`predict`).
  - Builds model, datasets, Lightning module, and datamodule.
- Core package: [src](src)
  - Models: [src/models.py](src/models.py)
  - Datasets: [src/datasets.py](src/datasets.py)
  - Lightning run module: [src/litrun_module.py](src/litrun_module.py)
  - Lightning data module: [src/litdata_module.py](src/litdata_module.py)

## Repo Conventions For Agents

- Keep Hydra override style consistent with Makefile examples, including `+dataset.*` keys when adding non-default fields.
- Use lower-risk defaults for edits to experiment commands:
  - preserve existing checkpoint paths unless explicitly asked to change them
  - preserve `run_name` patterns used by existing targets
- Do not modify large generated artifacts or experiment outputs unless explicitly requested:
  - [checkpoints](checkpoints)
  - [outputs](outputs)
  - [lightning_logs](lightning_logs)
  - dataset blobs under [data](data)
- Notebooks are used heavily for data prep and evaluation; if changing shared logic, prefer package modules under [src](src) and keep notebook edits minimal.

## Validation Expectations

- There is no established lint/test suite in this repo.
- For code changes, validate by running the most relevant existing Make target (or a minimal `python -u run.py ...` command copied from [Makefile](Makefile)).
- Report clearly if validation is skipped due to missing data files, GPU availability, or long-running workload constraints.

## Common Pitfalls

- `run.py` checks requested GPUs against `GPUtil.getAvailable(...)`; invalid GPU selections raise a hard error.
- Several workflows assume large HDF5 inputs already exist (for example under `data/pangolin_original`). Missing paths fail at dataset initialization.
- Stage-specific dataset requirements are strict in [run.py](run.py) and [src/datasets.py](src/datasets.py):
  - training expects `dataset.train_path`
  - eval/predict expect `dataset.predict_path`
- Some training/eval targets are designed for SLURM via `sbatch_redirect`; avoid changing scheduler behavior unless requested.
