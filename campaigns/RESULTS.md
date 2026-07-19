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
| `codex-6h-hybrid-10m-20260718-053618` | 600 s | fast body, stable head/scalars | 2.608312, 2.655714 | 2.632013 | Reject; early gain reverses late |
| same | 600 s | quarter learning rates | 2.434815, 2.443969 | 2.439392 | Retain stable incumbent |

Batch 8 used about 7.6 GB per lane, while batch 16 used about 14.5 GB. The
crossover is necessary because GPU 0 enters firmware thermal slowdown and is
typically slower than GPU 1. The smaller device batch nevertheless won the
mean validation result after swapping placements.

The half-rate recipe reached a fixed-probe loss near 6.81 before regressing
toward random loss in both assignments. Its output-head RMS grew roughly twice
as large as the quarter-rate recipe and its learned input-residual coefficient
crossed zero. A single best checkpoint or a single short run would therefore
have promoted the wrong configuration.

The targeted hybrid held output-head and scalar learning rates at the stable
values while doubling only embedding and matrix rates. It reproduced the same
misleading shape in both crossover placements: a substantially better probe
near the middle of training followed by regression during cooldown. Its paired
mean was 0.192621 worse than the stable control. The control mean differed by
only 0.000092 between the two independent 10-minute campaigns, which is a
useful check that the harness and validation path are repeatable.

## Correctness and paper-inspired smoke tests

| Campaign | Budget / lane | Treatment | Score | Outcome |
| --- | ---: | --- | ---: | --- |
| `codex-6h-swiglu-smoke-20260718-050206` | 60 s | parameter-matched SwiGLU | 2.891194 | Healthy; slightly faster, requires crossover |
| same | 60 s | ReLU-squared | 2.895492 | Control |
| `codex-6h-budgetfix-smoke-20260718-050544` | 60 s | half LR | 3.022318 | Confirmed eager work is counted |
| same | 60 s | quarter LR | 3.110100 | Confirmed eager work is counted |
| `codex-6h-fp32-smoke-20260718-053304` | 60 s | FP32 Adam state/master | 3.021920 | Healthy, about 160 MB extra, neutral screen |
| same | 60 s | native state | 3.020228 | Control |
| `codex-6h-swiglu-10m-20260718-060014` | 600 s | parameter-matched SwiGLU | 2.463151, 2.422131 | Mean 2.442641; reject |
| same | 600 s | ReLU-squared | 2.434733, 2.443788 | Mean 2.439260; retain |
| `codex-6h-amp-10m-20260718-062405` | 600 s | BF16 autocast | 2.009695, 1.995127 | Mean 2.002411; promote |
| same | 600 s | FP16 autocast | 2.434835, 2.443807 | Mean 2.439321; reject |
| `codex-6h-value-bf16-10m-20260718-065236` | 600 s | no value residuals | 1.989508, 2.003576 | Mean 1.996542; promote |
| same | 600 s | ResFormer value residuals | 2.007387, 1.994961 | Mean 2.001174; reject |
| `codex-6h-gated-smoke-20260718-071606` | 60 s | headwise gated attention | 3.065947 | Healthy; requires crossover |
| same | 60 s | standard attention | 3.060079 | Control |
| `codex-6h-gated-10m-20260718-071846` | 600 s | headwise gated attention | 2.010097, 1.996362 | Mean 2.003229; reject |
| same | 600 s | standard attention | 1.989243, 2.001488 | Mean 1.995366; retain |
| `codex-6h-attnres-smoke-20260719-073237` | 60 s | blockwise Attention Residuals | 3.158942 | Reject; 8 steps, 3.03 GB |
| same | 60 s | standard residuals | 3.057921 | Retain; 16 steps, 6.03 GB |

The budget-accounting fix produced 13 counted optimizer steps and roughly
60-62 seconds of measured training per lane. Before the fix, eager experiments
discarded 11 real optimizer steps as though they were compiler warm-up.

SwiGLU follows [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
with an approximately parameter-matched hidden dimension. FP32 Adam state and
master weights follow the numerical-safety idea in
[Mixed Precision Training](https://arxiv.org/abs/1710.03740). Neither smoke
result is treated as evidence of a model improvement by itself.

The full SwiGLU crossover demonstrated a hardware-dependent efficiency/quality
trade. SwiGLU completed 133 steps (69.73M tokens) on thermally constrained GPU
0 and 140 steps (73.40M tokens) on GPU 1, versus 128 and 123 steps for ReLU
squared. Its 2.442641 paired mean was nevertheless 0.003381 worse than ReLU
squared, and its 0.041020 spread was more than four times the control spread.
SwiGLU remains a tested optional implementation but is not the incumbent.

BF16 improved the paired mean by 0.436910 (17.9 percent relative), completed
138 and 145 steps versus FP16's 128 and 123, and used about 884 MB less peak
VRAM in both placements. The fixed-probe trajectories replicated closely and
ended at their best values. FP16 gradients stayed extremely small through the
early schedule before rising sharply; this unscaled mixed-precision loop was
discarding useful signal. BF16's wider exponent range preserved it and is now
the campaign default on this Ampere host.

Removing the ResFormer-inspired value residuals improved the paired mean by
0.004631 while reducing the model from 50.33M to 33.55M parameters. It also
saved about 193 MB of peak allocated VRAM and improved throughput by roughly
2 percent in both placements. The value-residual model learned slightly faster
per optimizer step, but the leaner model completed more useful steps inside the
fixed wall-clock budget and won the deciding metric. Value residuals are now
disabled in the default recipe.

Headwise gated attention follows the query-dependent post-SDPA gate from
[Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708).
It was healthy and nearly parameter-neutral, but its replicated mean was
0.007863 worse than standard attention and it used about 137 MB more peak
VRAM. The implementation remains available behind `ATTN_OUTPUT_GATE`, but is
not part of the incumbent.

Blockwise [Attention Residuals](https://arxiv.org/abs/2603.15031) ran correctly
through the `flash-attn-res` custom backward and cut peak memory roughly in
half. On these RTX 3090 lanes it stabilized near 100k tokens/s versus 130k for
standard residuals, completed only 8 versus 16 counted smoke steps, and scored
0.101020 worse. That efficiency gap makes a long crossover ineligible under
the fixed wall-clock objective.

## Promotion policy

- Promote by two-replica crossover mean, not the minimum score.
- Require the fixed-probe trajectory to remain stable through cooldown.
- Report throughput, parameter count, peak VRAM, and thermal placement.
- Reserve 30-60 minute lanes for the strongest survivor and a stable control.
