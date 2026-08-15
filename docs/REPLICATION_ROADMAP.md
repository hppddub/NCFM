# FT-NCFM Replication Roadmap

This repository starts from the official NCFM implementation at upstream commit
`9e3e60b855fa918337013c8c3d460601690eb58e` and adds an auditable FT-NCFM
reproduction layer. The target paper is arXiv:2511.16233v1 (20 November 2025).

The work is deliberately split into gates. A passing algorithmic smoke test is not
reported as a VLA benchmark reproduction.

## Paper claims to compare

| Claim | Paper target | Planned measurement |
| --- | ---: | --- |
| CALVIN, 5% coreset | Avg. Len 3.11 | Three-seed ABC-to-D evaluation |
| Meta-World, 5% coreset | Avg. success 50.5 +/- 0.8% | MT50 three-seed evaluation |
| LIBERO, 5% coreset | Avg. success 70.3 +/- 0.9% | Spatial/Object/Goal/Long mean |
| Contrastive verification value | CALVIN 3.11 vs. 2.81 without it | Paired ablation at fixed coreset size |
| End-to-end efficiency | More than 80% training-time reduction | Wall time, GPU-hours, peak memory |

## Phase 0 - immutable base and environment

- Keep `upstream` pointed at `gszfwsb/NCFM`.
- Develop on `codex/ft-ncfm-replication`.
- Record the upstream SHA, Python/PyTorch/CUDA versions, device, seed, config hash,
  and git SHA with every run.
- Gate: unit tests pass on CPU and a run manifest can be regenerated.

## Phase 1 - isolate NCFM

- Test the characteristic-function (CF) loss independently of CIFAR/ImageNet.
- Add optional real-sample weights to the real empirical characteristic function.
- Preserve the original uniform path bit-for-bit when no weights are supplied.
- Test the min-max sign convention: synthetic tensors minimize CF distance and
  `SampleNet` maximizes it.
- Gate: a deterministic two-dimensional mixture experiment reduces weighted CF
  distance and covers the high-weight modes.

## Phase 2 - FT influence engine

- Lightly train a guide policy for 15% of the full baseline step budget.
- Compute one validation gradient and approximate its inverse-Hessian-vector
  product with LiSSA (depth 50, damping 0.01, Hessian batch size 32).
- Score every training example using the paper's Eq. 2.
- Select the top 5% elite set.
- Generate minimal visual counterexamples while holding language and action fixed.
- Refine elite scores with Eq. 5 using beta 1.0.
- Convert raw scores to valid probabilities with a named, logged policy.
- Gate: analytical quadratic tests validate LiSSA and the refinement equation;
  all normalized weights are finite, non-negative, and sum to one.

## Phase 3 - 5% public-data proof of concept

- Use MNIST as a small public visual source and derive a VLA-shaped proxy task:
  image plus instruction token predicts a continuous action vector.
- Use a deterministic stratified 5% of the 60,000-example training split.
- Run the guide, influence, counterexample, and weighted NCFM stages end to end.
- Compare full proxy data, random 5%, influence-only 5%, uniform NCFM 5%, and
  FT-NCFM 5% using seeds 42, 123, and 1024.
- Gate: the adversarial game is numerically stable and its smoothed generator loss
  falls from the initial window to the final window. This validates plumbing only.

## Phase 4 - mini-VLA benchmark

- Replace the proxy adapter with LIBERO, initially one task and then one suite.
- Use the paper's ViT-B/16 visual backbone and six-layer Transformer policy when
  practical; record any smaller surrogate separately.
- Implement simulator-backed object substitution, size scaling, and position change.
- Run 1%, 5%, and 10% coresets plus all ablations.
- Gate: one task reproduces convergence and evaluation from a clean environment.

## Phase 5 - paper-scale comparison

- Run CALVIN ABC-to-D, Meta-World MT50, and all four LIBERO suites.
- Use seeds 42, 123, and 1024 and report mean, standard deviation, paired tests,
  GPU-hours, wall time, and peak device memory.
- Compare against the exact paper tables and retain raw per-seed artifacts.
- Gate: every table cell is traceable to a manifest, checkpoint, and metrics file.

## Reproducibility gaps and declared choices

The v1 paper does not specify the following. They must never be implicit:

1. **Influence sign.** Eq. 2 uses the classical leading minus sign, although the
   text later treats large positive values as helpful. The implementation exposes
   the sign policy and logs it.
2. **Non-negative sampling weights.** Eq. 5 can retain a negative base score, but
   Eq. 7 requires probabilities. The default is a positive shift plus epsilon;
   clamp and softmax alternatives are ablations.
3. **Reference gradient.** The paper refers to a relevant standard test case but
   does not define selection or aggregation. The default is the mean validation
   loss gradient over a fixed, stratified reference batch.
4. **LiSSA scaling and resampling.** Depth and damping are given, but scale,
   repeats, and batch-resampling rules are not. Every value is in the run config.
5. **Feature encoder and synthetic modality representation.** The paper gives a
   high-level ViT/Transformer description but no released FT-NCFM code. Proxy and
   benchmark adapters are therefore reported separately.
6. **Convergence criterion.** The pseudocode says "while not converged" without a
   rule. We use fixed steps plus a predeclared moving-window loss check.

## Compute boundary

The paper used one NVIDIA A100 80GB and reports a 24 GPU-hour FT plus NCFM
preprocessing cost on LIBERO. The current Windows workspace reports no NVIDIA GPU.
CPU execution is appropriate for unit tests and the public-data proxy, not for a
credible benchmark-scale performance comparison.
