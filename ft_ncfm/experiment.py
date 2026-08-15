"""Run the 5% public-data FT-NCFM algorithmic proof of concept."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from .config import config_hash, load_config
from .contrastive import ContrastiveVerifier
from .data import load_mnist_minivla
from .distillation import (
    WeightedNCFMDistiller,
    summarize_convergence,
    trace_as_dicts,
)
from .influence import InfluenceEngine, move_batch_to_device
from .models import MiniVLAPolicy


def policy_loss(model: nn.Module, batch: dict[str, Tensor]) -> Tensor:
    prediction = model(batch["image"], batch["instruction"])
    return F.mse_loss(prediction, batch["action"])


def set_seed(
    seed: int,
    *,
    deterministic: bool = False,
    deterministic_warn_only: bool = False,
) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if device.type != "cuda" or not torch.cuda.is_available():
        return metadata

    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    metadata.update(
        {
            "device_count": torch.cuda.device_count(),
            "device_index": index,
            "device_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "peak_allocated_memory_bytes": torch.cuda.max_memory_allocated(index),
            "peak_reserved_memory_bytes": torch.cuda.max_memory_reserved(index),
        }
    )
    try:
        metadata["driver_version"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        metadata["driver_version"] = None
    return metadata


def git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def maybe_limit_dataset(dataset: Dataset, maximum: int | None) -> Dataset:
    if maximum is None or len(dataset) <= maximum:
        return dataset
    return Subset(dataset, list(range(maximum)))


def train_guide(
    model: MiniVLAPolicy,
    dataset: Dataset,
    config: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    guide_config = config["guide"]
    loader = DataLoader(
        dataset,
        batch_size=int(guide_config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(guide_config["learning_rate"]),
        weight_decay=float(guide_config["weight_decay"]),
    )
    steps = max(
        1,
        math.ceil(int(guide_config["full_steps"]) * float(guide_config["duration_ratio"])),
    )
    iterator = iter(loader)
    metrics: list[dict[str, float | int | str]] = []
    model.train()
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = policy_loss(model, batch)
        loss.backward()
        optimizer.step()
        metrics.append({"stage": "guide", "step": step, "loss": float(loss.detach().cpu())})
    return metrics


@torch.no_grad()
def encode_dataset(
    model: MiniVLAPolicy,
    dataset: Dataset,
    *,
    device: torch.device,
) -> Tensor:
    model.eval()
    features: list[Tensor] = []
    for batch in DataLoader(dataset, batch_size=256, shuffle=False):
        batch = move_batch_to_device(batch, device)
        representation = model.encode(batch["image"], batch["instruction"])
        features.append(torch.cat([representation, batch["action"]], dim=1).cpu())
    return torch.cat(features)


def single_sample_batches(dataset: Dataset) -> list[dict[str, Tensor]]:
    return [batch for batch in DataLoader(dataset, batch_size=1, shuffle=False)]


def run_seed(
    config: dict[str, Any],
    *,
    seed: int,
    output_root: Path,
    device: torch.device,
    download: bool,
    max_samples: int | None,
) -> dict[str, Any]:
    runtime_config = config.get("runtime", {})
    deterministic = bool(runtime_config.get("deterministic", False))
    deterministic_warn_only = bool(runtime_config.get("deterministic_warn_only", False))
    set_seed(
        seed,
        deterministic=deterministic,
        deterministic_warn_only=deterministic_warn_only,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    data_config = config["data"]
    train_dataset, validation_dataset = load_mnist_minivla(
        str(data_config["cache_dir"]),
        train_ratio=float(data_config["train_ratio"]),
        validation_size=int(data_config["validation_size"]),
        seed=seed,
        download=download,
    )
    train_dataset = maybe_limit_dataset(train_dataset, max_samples)
    validation_dataset = maybe_limit_dataset(validation_dataset, 64 if max_samples else None)

    model = MiniVLAPolicy(hidden_dim=int(config["guide"]["hidden_dim"])).to(device)
    metrics = train_guide(model, train_dataset, config, seed=seed, device=device)
    model.eval()

    influence_config = config["influence"]
    reference_loader = DataLoader(
        validation_dataset,
        batch_size=int(influence_config["reference_batch_size"]),
        shuffle=False,
    )
    hessian_loader = DataLoader(
        train_dataset,
        batch_size=int(influence_config["hessian_batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    samples = single_sample_batches(train_dataset)
    influence_engine = InfluenceEngine(
        model,
        policy_loss,
        sign=str(influence_config["sign"]),
        device=device,
    )
    influence_result = influence_engine.score_samples(
        samples,
        reference_loader,
        hessian_loader,
        lissa_depth=int(influence_config["lissa_depth"]),
        lissa_damping=float(influence_config["lissa_damping"]),
        lissa_scale=float(influence_config["lissa_scale"]),
        lissa_repeats=int(influence_config["lissa_repeats"]),
    )

    verifier = ContrastiveVerifier(
        model,
        policy_loss,
        beta=float(influence_config["beta"]),
        elite_ratio=float(influence_config["elite_ratio"]),
        normalization=str(influence_config["normalization"]),
        epsilon=float(influence_config["epsilon"]),
    )
    contrastive_result = verifier.verify(
        influence_result.base_scores.to(device),
        samples,
        influence_result.reference_gradient,
    )
    probabilities = contrastive_result.normalized_weights.detach().cpu()
    real_features = encode_dataset(model, train_dataset, device=device)

    distillation_config = config["distillation"]
    synthetic_count = max(
        1, math.ceil(len(train_dataset) * float(distillation_config["coreset_ratio"]))
    )
    variant_summaries: dict[str, Any] = {}
    coresets: dict[str, Tensor] = {}
    for variant, weights in (("uniform", None), ("ft_ncfm", probabilities)):
        set_seed(
            seed + 10_000,
            deterministic=deterministic,
            deterministic_warn_only=deterministic_warn_only,
        )
        distiller = WeightedNCFMDistiller(
            real_features,
            weights,
            synthetic_count=synthetic_count,
            num_frequencies=int(distillation_config["num_frequencies"]),
            hidden_dim=int(distillation_config["hidden_dim"]),
            generator_lr=float(distillation_config["generator_lr"]),
            discriminator_lr=float(distillation_config["discriminator_lr"]),
            device=device,
        )
        trace = distiller.run(
            int(distillation_config["steps"]),
            discriminator_steps=int(distillation_config["discriminator_steps"]),
        )
        convergence = summarize_convergence(
            trace,
            window_ratio=float(distillation_config["convergence_window_ratio"]),
        )
        variant_summaries[variant] = asdict(convergence)
        coresets[variant] = distiller.coreset()
        for record in trace_as_dicts(trace):
            metrics.append({"stage": "distillation", "variant": variant, **record})

    torch.save(
        {
            "sample_ids": influence_result.sample_ids,
            "base_scores": influence_result.base_scores,
            "elite_indices": contrastive_result.elite_indices.cpu(),
            "original_scores": contrastive_result.original_scores.cpu(),
            "counterexample_scores": contrastive_result.counterexample_scores.cpu(),
            "raw_refined_weights": contrastive_result.raw_refined_weights.cpu(),
            "normalized_probabilities": probabilities,
            "normalization_policy": influence_config["normalization"],
            "templates": contrastive_result.template_names,
        },
        seed_dir / "influence.pt",
    )
    torch.save(
        {
            "schema": "joint_feature_proxy",
            "synthetic_count": synthetic_count,
            "variants": coresets,
        },
        seed_dir / "coreset.pt",
    )
    write_jsonl(seed_dir / "metrics.jsonl", metrics)

    elapsed = time.perf_counter() - start
    summary = {
        "seed": seed,
        "source_train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "synthetic_samples": synthetic_count,
        "declared_source_fraction_of_corpus": float(data_config["train_ratio"]),
        "declared_synthetic_fraction_of_source": float(distillation_config["coreset_ratio"]),
        "declared_synthetic_fraction_of_corpus": round(
            float(data_config["train_ratio"]) * float(distillation_config["coreset_ratio"]),
            12,
        ),
        "debug_max_samples": max_samples,
        "elite_samples": int(contrastive_result.elite_indices.numel()),
        "variants": variant_summaries,
        "elapsed_seconds": elapsed,
    }
    manifest = {
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "git_sha": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "upstream_sha": git_output("rev-parse", "upstream/main"),
        "config": config,
        "config_sha256": config_hash(config),
        "seed": seed,
        "device": str(device),
        "deterministic_algorithms": deterministic,
        "deterministic_warn_only": deterministic_warn_only,
        "runtime_overrides": {"download": download, "max_samples": max_samples},
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cuda": cuda_metadata(device),
        "execution": {
            "provider": os.getenv("FT_NCFM_EXECUTION_PROVIDER", "local"),
            "runtime_id": os.getenv("FT_NCFM_RUNTIME_ID"),
        },
        "colab": {
            "release_tag": os.getenv("COLAB_RELEASE_TAG"),
            "backend_version": os.getenv("COLAB_BACKEND_VERSION"),
            "runtime_version": os.getenv("COLAB_RUNTIME_VERSION"),
        },
        "aws": {
            "ami_id": os.getenv("FT_NCFM_AWS_AMI_ID"),
            "region": os.getenv("FT_NCFM_AWS_REGION"),
            "availability_zone": os.getenv("FT_NCFM_AWS_AVAILABILITY_ZONE"),
            "instance_id": os.getenv("FT_NCFM_AWS_INSTANCE_ID"),
            "instance_type": os.getenv("FT_NCFM_AWS_INSTANCE_TYPE"),
            "max_runtime_minutes": os.getenv("FT_NCFM_AWS_MAX_RUNTIME_MINUTES"),
            "stack_name": os.getenv("FT_NCFM_AWS_STACK_NAME"),
        },
        "container": {
            "image": os.getenv("FT_NCFM_CONTAINER_IMAGE"),
            "base_image_digest": os.getenv("FT_NCFM_BASE_IMAGE_DIGEST"),
        },
        "packages": {
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
            "numpy": package_version("numpy"),
        },
    }
    write_json(seed_dir / "summary.json", summary)
    write_json(seed_dir / "manifest.json", manifest)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/ft_ncfm/minivla_5pct.yaml", help="YAML config path"
    )
    parser.add_argument("--offline", action="store_true", help="Do not download MNIST")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Debug-only cap after ratio selection; never use for reported results",
    )
    parser.add_argument("--seed", type=int, default=None, help="Run one seed instead of all")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override experiment.output_dir while retaining it in the run manifest",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = copy.deepcopy(load_config(args.config))
    if args.output_dir is not None:
        config["experiment"]["output_dir"] = args.output_dir
    seeds = [args.seed] if args.seed is not None else list(config["experiment"]["seeds"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(config["experiment"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = [
        run_seed(
            config,
            seed=int(seed),
            output_root=output_root,
            device=device,
            download=not args.offline,
            max_samples=args.max_samples,
        )
        for seed in seeds
    ]
    write_json(output_root / "aggregate_summary.json", {"runs": summaries})
    print(json.dumps({"output": str(output_root), "runs": summaries}, indent=2))


if __name__ == "__main__":
    main()
