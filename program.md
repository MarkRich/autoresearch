# Autoresearch Program: Paired 30-Minute Transformer Campaign

This campaign runs on `lambda-quad`, using its two RTX 3090s as independent
single-GPU lanes. The research target is the causal Transformer implementation
in `train.py`; the deciding metric remains validation bits per byte (`val_bpb`).

## Integrity rules

- Do not edit `prepare.py`, its validation data, or `evaluate_bpb`.
- Give each run exactly 1,800 seconds of measured training time.
- Send the same experiment to both GPUs with seeds 101 and 202.
- Compare paired means and the spread between replicas. Never promote the
  lowest isolated score from one lucky run.
- Use `WINDOW_PATTERN=LLLL` with the PyTorch SDPA fallback. Sliding-window
  results are invalid unless FA3 is active and actually honors the window.
- Keep dashboard failure non-fatal; local JSONL metrics are the source of truth.
- Record campaign id, run id, code SHA, seed, GPU lane, environment, and exact
  hyperparameters with every result.
- Stop after the finite six-round plan. This is not an unattended forever loop.

## Default plan

`campaigns/run_campaign.py` owns the reproducible plan:

1. conservative Muon control
2. remove value embeddings
3. reduce matrix learning rate to 0.0015
4. increase matrix learning rate to 0.0025
5. extend warmdown to 70%
6. decay the final learning rate to zero

Every item gets two replicas. Six concurrent pairs at 30 minutes each require
about three hours of training, plus startup and fixed validation passes.

## Running

The production campaign uses the existing prepared dataset and Python runtime:

```bash
cd /home/mark/worktrees/autoresearch-30m
CODEX_WORK_TAG=autoresearch-30m-campaign \
  /home/mark/autoresearch-transformer-paper/.venv/bin/python -u \
  campaigns/run_campaign.py
```

Run a one-round 60-second preflight before production changes. The preflight
must finish both replicas, produce final `val_bpb` events, return the GPUs to
idle, and leave no worker process behind.

## Interpreting results

Lower `val_bpb` is better. Treat a change as promising only when the paired
result is directionally credible, the run completed its full budget, and its
fixed-probe loss does not show catastrophic instability. A tiny improvement
smaller than the replica spread is a follow-up candidate, not a conclusion.

Never print or commit dashboard credentials. They live in the mode-0600 file
`~/.config/autoresearch-dashboard.env` on the GPU host.
