# Final Independent Audit

Audit date: 15 August 2026 (America/Toronto)

Scope: branch through `3aa132839ef98917fe61fb115427cfb98a2e2cdf` and the
exported Colab archive. The auditor made no file changes. Documentation being
written concurrently was excluded from the commit-scoped assessment.

## Verdict

No P0 defects were found. The A100 execution and evidence chain are valid for a
deterministic MNIST feature-proxy experiment. They do not show that FT-NCFM
outperforms uniform NCFM and are not comparable with the paper's VLA benchmark
findings.

## Independently verified

- Archive SHA-256:
  `13e688c57cba1cc495c292970cbaf9fd93ec293db78f0cb09ce03fa530859225`.
- Test suite: 28 passed; Ruff: all checks passed.
- The deterministic `AvgPool2d` replacement is exactly equal to the former
  fixed-shape adaptive pooling in both forward values and input gradients.
- The resume aggregation fix recovers seeds 42, 123, and 1024, and its aggregate
  matches all three persisted summaries and recomputed metrics.
- All stored tensors load safely, are finite, and have expected shapes: 60,000
  influence entries, 3,000 elites, and 3,000-by-35 synthetic proxy tensors per
  true-5% seed. Probabilities sum to one and sample IDs are unique.
- Manifests record the A100-SXM4-80GB, compute capability 8.0, driver 580.82.07,
  pinned PyTorch/torchvision versions, deterministic algorithms with warnings
  disabled, clean experiment commit `01611704...`, and upstream NCFM commit
  `9e3e60b...`.
- Experiment computation occurred at `01611704...`; `3aa1328...` only regenerated
  aggregation across separately persisted invocations. The artifact manifests
  preserve this distinction.

## Severity-ranked findings

### P1 - no FT-NCFM superiority result

All six true-5% adversarial runs converged. FT-NCFM's raw final loss was 140.65%
higher at seed 42, 3.54% higher at seed 123, and 21.64% lower at seed 1024. Mean
raw final loss was 0.030949 for FT-NCFM and 0.022902 for uniform NCFM.

These objectives use different weighted target distributions, so the raw losses
are not an apples-to-apples method-quality metric. The defensible conclusion is
only that both objectives converged. Identical downstream policies and a held-out
task metric are required for comparison.

### P1 - algorithmic feature proxy, not paper replication

The executable runs only uniform and FT-weighted variants and saves joint feature
proxies rather than reusable `(vision, language, action)` samples. It lacks
simulator-grounded counterexamples, semantic instruction selection, downstream
VLA training, the complete declared baseline matrix, and CALVIN, LIBERO, or
Meta-World evaluation.

Peak allocation was roughly 0.16 GiB, and the three true-5% runs totaled about
0.242 GPU-hours. Those measurements cannot be compared with the paper's VLA
preprocessing cost, success rates, or training-time reduction.

### P1 - influence sign and ranking remain ambiguous

The true-5% config uses the paper-sign Eq. 2 score, positive-shift normalization,
and top-score selection. Under classical influence semantics this may prioritize
samples that increase reference loss. A preregistered factorial ablation over
sign, ranking direction, and normalization is required before an expensive VLA
run.

### P1 - reference and final evaluation are not isolated

The proxy uses the official MNIST test split as its influence reference. It must
not also be treated as an untouched downstream evaluation set. Future runs need
separate training, influence-reference validation, and final test partitions.

### P2 - persisted-seed validation was too weak

At the audited commit, resume checks required filenames only and aggregation did
not verify common config, executable commit, provider, or expected seed set. The
current post-audit runner now adds per-file SHA-256 checksums, parses and validates
JSON/JSONL/PT artifacts and finite tensors, verifies seed identity, and rejects
incompatible or undeclared seeds. Regression tests cover corruption and
incompatible manifests.

A first remediation review found three remaining P2 edge cases: JSON numeric
finiteness, comparison against the current requested run rather than only other
stored seeds, and incomplete runtime-override comparison. The runner now rejects
non-finite JSON/JSONL numbers, derives an expected identity from the active config,
Git state, provider, full runtime overrides, device, and GPU before any billable
seed work, and compares every persisted seed against it. The downloaded A100
archive remains a documented legacy bundle protected by its outer ZIP checksum;
it is not silently upgraded to the new per-seed checksum format.

The final narrow remediation audit caught and then verified a config-identity
edge case: the child overrides `experiment.output_dir` with its scratch path
before hashing. The runner now deep-copies the declared config, applies that exact
override before deriving the expected identity, and keeps seed declarations from
the untouched original. The auditor confirmed the fix with 18 focused tests and
clean Ruff output and found no remaining remediation defect. The persistence P2
findings are resolved for future runs.

### P2 - supporting gates have one seed only

The 1,000-record benchmark and nested smoke each have seed 42 only. Only the
true-5% gate has seeds 42, 123, and 1024. No multi-seed claim is made for the two
supporting runs.

## Audit disposition

Accept the work as an A100 infrastructure, determinism, persistence, and
convergence proof. Do not label it a replication of the paper's benchmark
findings. The next scientific gate is downstream held-out evaluation with the
influence ablations above and an actual trainable multimodal coreset.
