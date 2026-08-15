# Experiment Protocol

## Locked defaults from the paper

- Seeds: 42, 123, 1024
- Coreset ratios: 1%, 5%, 10%; 5% is the first proof of concept
- Guide duration: 15% of the standard training procedure
- Elite ratio: 5%
- Contrastive beta: 1.0
- LiSSA recursion depth: 50
- LiSSA damping: 0.01
- LiSSA Hessian batch size: 32
- Downstream optimizer: AdamW, learning rate 1e-4, weight decay 1e-2

## Required comparisons

1. Full-data policy
2. Random subset at the target ratio
3. Influence-function selection at the target ratio
4. Uniform NCFM at the target ratio
5. Influence-weighted NCFM without contrastive verification
6. Full FT-NCFM

All comparisons share data preprocessing, architecture, optimizer, step budget,
evaluation cases, and seed. Changing a surrogate architecture creates a new block
of results rather than overwriting the paper-comparison block.

## Artifact contract

Every run writes:

- `manifest.json`: git SHA, upstream SHA, config and config hash, host, package
  versions, device, seed, start/end timestamps;
- `metrics.jsonl`: one record per training/evaluation step;
- `influence.pt`: sample IDs, base scores, contrastive scores, raw refined weights,
  normalized probabilities, and normalization policy;
- `coreset.pt`: synthetic tensors and modality schema;
- `summary.json`: initial/final moving-window losses and evaluation metrics.

## Convergence decision

For the smoke test, let `g_t` be generator CF loss. A run passes if all losses are
finite and the median of the last 10% of steps is below the median of the first
10%. Benchmark conclusions require downstream success metrics; convergence alone
is not evidence that the paper's performance claims were reproduced.

## Ratio accounting

`data.train_ratio` is the fraction of the original real corpus supplied to the
distiller. `distillation.coreset_ratio` is the synthetic fraction of that supplied
source. Their product is the synthetic fraction of the original corpus and is
recorded in every summary. The quick nested smoke uses `0.05 * 0.05 = 0.0025`;
the paper-style 5% proxy uses `1.0 * 0.05 = 0.05`.
