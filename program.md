# Autoresearch Program: Transformer Paper Reimplementation

You are running on `lambda-quad`, a Linux workstation with NVIDIA GPUs. Your job is to use the
Karpathy autoresearch loop to reimplement and study the core ideas from "Attention Is All You Need"
inside this repo's compact single-file language-model training setup.

## Research Goal

Get a working, measurable Transformer-paper-inspired model in `train.py`, then improve validation
bits per byte (`val_bpb`) under the fixed five-minute training budget.

The original 2017 paper is an encoder-decoder sequence transduction model. This repo is a causal
language model, so do not force a full translation system into it. Instead, make the causal model
faithfully expose and test the paper's architectural ideas where they make sense:

- scaled dot-product multi-head attention
- learned token embeddings plus an explicit positional signal
- residual streams around attention and feed-forward sublayers
- layer normalization around sublayers
- position-wise feed-forward networks
- dropout or regularization only if it helps the five-minute metric
- clear, simple hyperparameters that make the implementation easy to inspect

The first useful milestone is not novelty. It is a clean, reviewable baseline that looks like the
paper's Transformer adapted to next-token prediction and trains successfully on the local GPU.

## Files

Read the repo before editing:

- `README.md` for the autoresearch rules.
- `prepare.py` for constants, data loading, tokenizer, and evaluation. Do not modify it.
- `train.py` for the editable model, optimizer, and training loop.

Only edit `train.py` unless you are fixing local logging around the experiment. Do not edit
`prepare.py` or the evaluation function.

## Setup

Use `uv` from the user install:

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/mark/autoresearch-transformer-paper
uv sync
uv run prepare.py
```

If full preparation is already complete, do not redo unnecessary work. The data/tokenizer cache is
under `~/.cache/autoresearch/`.

Initialize `results.tsv` if it does not exist:

```text
commit	val_bpb	memory_gb	status	description
```

## Telemetry

`train.py` emits local JSON telemetry into `metrics/`:

- `metrics/<run_id>.jsonl`
- `metrics/latest.json`

If the environment contains `DASHBOARD_INGEST_URL`, it also POSTs progress events to the dashboard.
Do not print secrets or tokens. If the dashboard endpoint fails, keep training; local metrics are the
source of truth.

Suggested run environment:

```bash
export AUTORESEARCH_RUN_ID="transformer-paper-$(date +%Y%m%d-%H%M%S)"
uv run train.py > "run-$AUTORESEARCH_RUN_ID.log" 2>&1
```

## Experiment Loop

1. Check git state and current commit.
2. Establish a baseline if none exists: run current `train.py` unchanged and log the result.
3. Make one coherent Transformer-paper-inspired change in `train.py`.
4. Commit the change.
5. Run `uv run train.py > run.log 2>&1`.
6. Extract the final summary with `grep "^val_bpb:\\|^peak_vram_mb:\\|^num_steps:" run.log`.
7. Append one row to `results.tsv`.
8. Keep the commit if `val_bpb` improves. If it is worse or crashes, reset back to the previous good commit.

Status values:

- `keep`: improvement or valuable baseline
- `discard`: valid run but worse metric
- `crash`: failed run or timeout

Each experiment should finish in roughly five minutes plus startup/evaluation. If a run exceeds ten
minutes, kill it, log it as a crash, and move on.

## First Research Directions

Start simple and paper-faithful:

- Compare the existing RoPE/RMSNorm/ResFormer-ish setup against a cleaner sinusoidal or learned
  positional embedding plus LayerNorm-style block.
- Try a straightforward attention + FFN block before adding optimizations back.
- Preserve efficient flash attention if it keeps the paper-like attention semantics and avoids slow
  runs on RTX 3090.
- Keep parameter count and VRAM visible. The metric matters, but the implementation should remain
  understandable.

Never dump secrets or inspect private credential files. Continue autonomously once the first run is
started.
