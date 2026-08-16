# Environment

## Paper reference environment

The FT-NCFM v1 appendix reports Ubuntu 22.04.3, Python 3.12.7, PyTorch 2.5.0,
torchvision 0.20.0, CUDA 12.4, one NVIDIA A100-SXM4-80GB, 16 CPU cores, 47 GiB
memory, and 196 GB storage.

The `Dockerfile` pins the paper's PyTorch/CUDA family. Image digests should be
recorded in benchmark manifests because a mutable tag is not an immutable runtime.

## Current local discovery

- Host: Windows
- Python default: 3.14.5 (not used for the pinned reproduction)
- Reproduction Python: 3.11.9 through `.venv`
- NVIDIA runtime: not detected
- WSL: not installed

This host can run CPU unit tests and the public-data proxy. The CALVIN,
Meta-World, and LIBERO stages require a separate Linux CUDA machine. Hardware
results from that machine must be added to the generated manifest; do not combine
its timings with local CPU timings.

## Colab execution

The hosted A100 path is documented in `COLAB_A100_RUNBOOK.md`. The checked-in
notebook verifies the allocation before installing dependencies and persists each
completed seed to Google Drive. Colab GPU type, memory capacity, and session length
are not fixed properties of this repository, so the generated manifest is the
source of truth for every run.
