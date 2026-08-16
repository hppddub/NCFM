# Nested-Fraction Public-Data Smoke Results

Run date: 15 August 2026 (America/Toronto)

These results validate the FT-NCFM implementation path and adversarial-game
convergence on CPU. They do **not** reproduce the CALVIN, Meta-World, or LIBERO
success-rate findings in the paper.

An independent audit identified that this historical run applied a 5% source
fraction and then a 5% synthetic fraction. It is therefore a 0.25%-of-corpus
nested smoke test, not a paper-comparable 5% coreset. The historical ratio
structure is now explicit in `configs/ft_ncfm/minivla_nested_5pct_smoke.yaml`; the corrected
`configs/ft_ncfm/minivla_5pct.yaml` uses the full source and produces 3,000
synthetic feature proxies.

## Setup

- Public source: MNIST training split
- Source fraction: exactly 5% (3,000 of 60,000), deterministically stratified
- VLA-shaped adapter: digit image plus instruction token predicts a three-value
  continuous action derived from visual center-of-mass and density
- Guide training: 30 steps, equal to 15% of a 200-step proxy baseline
- LiSSA: depth 50, damping 0.01, scale 10, Hessian batch 32
- Elite ratio: 5% (150 samples), beta 1.0
- Synthetic coreset: 150 joint feature tensors (5% of the 3,000-sample source)
- NCFM game: 150 steps, one discriminator update per generator update
- Seeds: 42, 123, 1024
- Convergence gate: finite losses and final-window median below initial-window median

## Generator CF loss

| Seed | Uniform initial | Uniform final | FT-NCFM initial | FT-NCFM final |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.071353 | 0.020958 | 0.057544 | 0.010520 |
| 123 | 0.062508 | 0.030174 | 0.099268 | 0.030761 |
| 1024 | 0.072937 | 0.021524 | 0.062096 | 0.015997 |
| **Mean** | **0.068933** | **0.024219** | **0.072969** | **0.019093** |

- All six uniform/weighted runs passed the convergence gate.
- Mean windowed loss reduction was 64.9% for uniform NCFM and 73.8% for FT-NCFM.
- FT-NCFM had the lower final loss in seeds 42 and 1024; uniform NCFM was slightly
  lower in seed 123. The FT-NCFM three-seed mean final loss was 21.2% lower.

The proper conclusion is narrow: the implemented influence-aware path is stable
and converges in the public-data proxy. These losses are optimization diagnostics,
not task success metrics, and cannot substantiate the paper's reported 85-90%
performance recovery or greater-than-80% training-time reduction.

## Next comparison gate

The next credible step is a single LIBERO task on Linux/CUDA with simulator-backed
object substitution, size scaling, and position change. Required comparisons are
full data, random 5%, influence selection 5%, uniform NCFM 5%, influence-only
weighted NCFM, and full FT-NCFM, all at seeds 42, 123, and 1024.
