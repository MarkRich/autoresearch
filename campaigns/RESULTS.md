# Autoresearch campaign record

Host: `lambda-quad` (2 x RTX 3090, independent one-process lanes)  
Primary metric: validation bits per byte (`val_bpb`, lower is better)  
Campaign start: 2026-07-18 UTC

This is a decision log, not a leaderboard of isolated lucky runs. Crossover
means are the promotion signal; one-round smoke tests only establish that a
code path is safe enough to measure.

## Calibration and stability

| Campaign | Budget / lane | Treatment | Scores | Mean | Decision |
| --- | ---: | --- | --- | ---: | --- |
| `codex-6h-preflight-20260718-044525` | 60 s | control | 2.935938, 2.933879 | 2.934908 | Harness healthy on both GPUs |
| `codex-6h-throughput-20260718-044950` | 180 s | device batch 8 | 2.443832, 2.476260 | 2.460046 | Promote |
| same | 180 s | device batch 16 | 2.518949, 2.472028 | 2.495488 | Reject; worse mean and more VRAM |
| `codex-6h-lrscale-10m-20260718-050849` | 600 s | half learning rates | 3.009350, 3.037510 | 3.023430 | Reject; repeatable late collapse |
| same | 600 s | quarter learning rates | 2.434727, 2.443874 | 2.439300 | Stable incumbent |

Batch 8 used about 7.6 GB per lane, while batch 16 used about 14.5 GB. The
crossover is necessary because GPU 0 enters firmware thermal slowdown and is
typically slower than GPU 1. The smaller device batch nevertheless won the
mean validation result after swapping placements.

The half-rate recipe reached a fixed-probe loss near 6.81 before regressing
toward random loss in both assignments. Its output-head RMS grew roughly twice
as large as the quarter-rate recipe and its learned input-residual coefficient
crossed zero. A single best checkpoint or a single short run would therefore
have promoted the wrong configuration.

## Correctness and paper-inspired smoke tests

| Campaign | Budget / lane | Treatment | Score | Outcome |
| --- | ---: | --- | ---: | --- |
| `codex-6h-swiglu-smoke-20260718-050206` | 60 s | parameter-matched SwiGLU | 2.891194 | Healthy; slightly faster, requires crossover |
| same | 60 s | ReLU-squared | 2.895492 | Control |
| `codex-6h-budgetfix-smoke-20260718-050544` | 60 s | half LR | 3.022318 | Confirmed eager work is counted |
| same | 60 s | quarter LR | 3.110100 | Confirmed eager work is counted |
| `codex-6h-fp32-smoke-20260718-053304` | 60 s | FP32 Adam state/master | 3.021920 | Healthy, about 160 MB extra, neutral screen |
| same | 60 s | native state | 3.020228 | Control |

The budget-accounting fix produced 13 counted optimizer steps and roughly
60-62 seconds of measured training per lane. Before the fix, eager experiments
discarded 11 real optimizer steps as though they were compiler warm-up.

SwiGLU follows [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
with an approximately parameter-matched hidden dimension. FP32 Adam state and
master weights follow the numerical-safety idea in
[Mixed Precision Training](https://arxiv.org/abs/1710.03740). Neither smoke
result is treated as evidence of a model improvement by itself.

## Promotion policy

- Promote by two-replica crossover mean, not the minimum score.
- Require the fixed-probe trajectory to remain stable through cooldown.
- Report throughput, parameter count, peak VRAM, and thermal placement.
- Reserve 30-60 minute lanes for the strongest survivor and a stable control.
