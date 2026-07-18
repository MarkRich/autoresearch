# Autoresearch Program: Independent-GPU Transformer Campaign

This campaign runs on `lambda-quad`, using its two RTX 3090s as independent
single-GPU lanes. The research target is the causal Transformer in `train.py`;
the deciding metric is validation bits per byte (`val_bpb`, lower is better).

## Integrity rules

- Do not edit `prepare.py`, its validation data, or `evaluate_bpb`.
- Run one process per GPU. Never use a result from a shared multi-GPU job.
- Screen changes cheaply, then confirm survivors with two-round crossover plans:
  each treatment uses both GPUs and both seeds (101 and 202).
- Rank treatments by replicated mean, not the best isolated score. Report the
  spread so improvements smaller than run-to-run variation remain provisional.
- Keep the training-time budget honest. Eager runs have no uncounted warm-up;
  compiled runs explicitly declare the compiler warm-up steps they exclude.
- Use `WINDOW_PATTERN=LLLL` with the PyTorch SDPA fallback. Sliding-window
  results are invalid unless FA3 is active and actually honors the window.
- Record campaign id, run id, code SHA, seed, GPU lane, resolved environment,
  parameter count, measured training seconds, token count, peak VRAM, and exact
  hyperparameters with every result.
- Stop a lane if the GPU reaches 88 degrees C, the loss becomes non-finite, or
  the fixed-probe regression guard detects a sustained collapse.
- Keep dashboard failure non-fatal; local JSONL metrics are the source of truth.

## Search strategy

The campaign is a funnel rather than an open-ended random search:

1. **Harness calibration:** validate two independent lanes, honest budgets,
   batch-size throughput, and thermal safety.
2. **Stability:** find a learning-rate recipe that does not improve and then
   collapse late in the run. Diagnose the embedding, output head, residual
   coefficients, and gradient norm before changing architecture.
3. **Low-cost screens:** use 60-180 second runs only for correctness, memory,
   and large throughput or optimization signals.
4. **Crossover confirmation:** give credible treatments 5-10 minutes per lane,
   swap GPU/seed assignments, and promote by paired mean.
5. **Paper-inspired architecture:** test parameter-matched SwiGLU from
   [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202), value
   residuals from [ResFormer](https://arxiv.org/abs/2410.17897), and blockwise
   [Attention Residuals](https://arxiv.org/abs/2603.15031), plus headwise output
   gating from [Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708),
   only on the stable optimizer recipe.
6. **Numerical efficiency:** compare FP16 with BF16 using the same stable
   recipe, motivated by BF16's FP32-like exponent range in
   [A Study of BFLOAT16 for Deep Learning Training](https://arxiv.org/abs/1905.12322).
7. **Long confirmation:** spend 30-60 minutes per lane only on the strongest
   survivor and its stable control.

BF16 is the current numerical incumbent after a placement-swapped 10-minute
crossover. Before committing an hour to one recipe, compare its quarter-scale
learning rates with an eighth-scale alternative over 30 minutes per lane. This
longer horizon tests whether the faster BF16 learning signal needs a lower peak
rate to preserve late-run stability.

Mixed-precision master weights are available as a diagnostic based on
[Mixed Precision Training](https://arxiv.org/abs/1710.03740), but are not
promoted merely because they are theoretically safer; they must improve the
paired validation result enough to justify their memory and copy overhead.

## Running a plan

The campaign runner accepts explicit JSON plans. Each round must contain
exactly one lane for GPU 0 and one for GPU 1. For example:

```bash
cd /home/mark/worktrees/autoresearch-campaign
CODEX_WORK_TAG=autoresearch-6h \
  /home/mark/autoresearch-transformer-paper/.venv/bin/python -u \
  campaigns/run_campaign.py \
    --budget-seconds 600 \
    --plan-file campaigns/plans/hybrid-lr-crossover.json \
    --campaign-id autoresearch-hybrid-10m
```

Before a new code path receives a crossover, run one short lane per GPU and
require both workers to produce final events, return code zero, and release
their GPU memory. Summarize a completed campaign with:

```bash
python campaigns/summarize_campaign.py campaign-results/<campaign-id>
```

## Promotion standard

A candidate is eligible for a long run only when all of the following hold:

- both crossover assignments finish their full measured training budget;
- its paired mean beats the control, rather than relying on one lucky seed;
- the improvement is credible relative to the between-replica spread;
- fixed-probe loss and diagnostic signals do not show late collapse;
- parameter count, peak memory, and throughput costs are included in the
  decision rather than hidden behind the validation score.

Never print or commit dashboard credentials. They live in the mode-0600 file
`~/.config/autoresearch-dashboard.env` on the GPU host.
