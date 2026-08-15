# Colab A100 Runbook

This is the preferred compute path while the AWS accelerated-instance quota is
unavailable. The AWS stack remains deployed with `LaunchInstance=false`; this
runbook does not use AWS credentials or AWS resources.

## Scope and evidence boundary

The notebook runs the MNIST Mini-VLA feature-proxy experiment. It can establish
that the influence, contrastive-verification, and min-max distillation plumbing
runs on CUDA and whether its declared convergence rule passes. It cannot establish
the FT-NCFM paper's CALVIN, Meta-World, or LIBERO performance claims.

The independent audit remains controlling: a paper-comparable run is blocked by
the influence-sign ambiguity, missing untouched evaluation split, feature-level
rather than `(V,L,A)` coreset, proxy rather than simulator-grounded
counterexamples, and missing downstream baselines.

## Start a clean Colab session

1. Open the [FT-NCFM A100 notebook](https://colab.research.google.com/github/hppddub/NCFM/blob/codex/ft-ncfm-replication/notebooks/FT_NCFM_Colab_A100.ipynb).
2. In Colab, choose **Runtime -> Change runtime type** and select **A100 GPU**.
3. Run the settings and hardware cells. Stop if the hardware cell does not report
   an NVIDIA A100. Record whether the allocation is 40 GB or 80 GB; only an 80 GB
   result is close to the paper's A100-SXM4-80GB memory capacity.
4. Mount Google Drive when prompted. The notebook writes only below
   `MyDrive/FT-NCFM`.
5. Run the clone, dependency, test, and preflight cells in order. The preflight
   writes a session manifest to Drive without starting the experiment.

The notebook installs PyTorch 2.5.0 and torchvision 0.20.0 from the CUDA 12.4
wheel index when necessary. It then prints and records the actual PyTorch, CUDA,
driver, cuDNN, GPU, Colab, Python, and Git versions. Do not describe a run as
environment-matched if those recorded values differ from the paper reference in
`docs/ENVIRONMENT.md`.

## Compute gates

Run these cells in order. Do not use **Run all** after enabling an expensive cell.

1. **1,000-sample benchmark:** enabled by default, seed 42. It verifies CUDA
   execution and provides the first realistic timing measurement.
2. **Nested smoke:** enabled by default, seed 42. It uses 3,000 source examples
   and 150 feature proxies, or 0.25% of the original corpus. This remains a
   plumbing-only experiment.
3. **True 5% seed 42:** disabled by default. It uses all 60,000 source examples
   and 3,000 feature proxies. Enable it only if both preceding runs finish, all
   losses are finite, both uniform and FT-NCFM variants satisfy the declared
   convergence check, and the projected session time is comfortably below the
   remaining Colab allocation.
4. **Seeds 123 and 1024:** disabled by default. Enable them only after seed 42's
   manifests and artifacts have been reviewed.

If the 1,000-sample benchmark takes more than 15 minutes, do not start the true
5% run. Optimize or batch the per-sample gradient path first. The complete run
currently performs one gradient calculation per source sample, and notebook
availability is not a scientific checkpointing mechanism.

## Persistence and resume behavior

Experiments run from `/content/ft-ncfm-scratch` for local VM speed. After a seed
finishes, the runner copies its five required artifacts to a `.partial` directory
on Drive and atomically renames it to `seed_<seed>`. On a rerun, a seed is skipped
only when all five artifacts exist:

- `manifest.json`
- `summary.json`
- `metrics.jsonl`
- `influence.pt`
- `coreset.pt`

Session manifests and console logs are written directly to Drive. A Colab
disconnect during a seed can therefore lose that seed's unfinished scratch work,
but it cannot cause a partial seed to be mistaken for a completed result. Restart
the notebook and rerun the same command to resume at the next incomplete seed.

Expected Drive layout:

```text
MyDrive/FT-NCFM/
  a100-preflight/sessions/
  benchmark-1000/
    sessions/
    logs/
    seed_42/
    aggregate_summary.json
  nested-smoke/
  true-5pct/
```

## Acceptance checks

For every completed seed:

- `manifest.json` reports `execution.provider` as `colab`, the expected Git SHA,
  `device: cuda`, the assigned A100 model, and the pinned package versions.
- `summary.json` reports the expected source and synthetic counts.
- Both `variants.uniform` and `variants.ft_ncfm` have finite losses.
- A plumbing convergence pass requires `final_median < initial_median`; downstream
  task claims still require the unimplemented evaluation block.
- The Drive directory contains the exact raw artifacts, not only copied console
  output.

After the first true 5% seed completes, stop compute and perform the planned
result review before launching the other two seeds.
