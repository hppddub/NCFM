"""Run a genuine one-task LIBERO FT-NCFM proof of concept."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import shutil
import time
import urllib.request
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from .config import config_hash, load_config
from .contrastive import ContrastiveVerifier
from .distillation import WeightedNCFMDistiller, summarize_convergence, trace_as_dicts
from .downstream import train_action_regressor
from .experiment import (
    cuda_metadata,
    git_output,
    package_version,
    set_seed,
    single_sample_batches,
    write_json,
    write_jsonl,
)
from .influence import InfluenceEngine, move_batch_to_device
from .libero_data import (
    LiberoHDF5Dataset,
    inspect_libero_hdf5,
    sha256_file,
    split_demo_keys,
)
from .libero_models import LiberoVLAPolicy, libero_policy_loss
from .libero_rollout import evaluate_libero_checkpoint


def _ensure_dataset(data: dict[str, Any], *, download: bool) -> Path:
    path = Path(str(data["path"])).expanduser().resolve()
    expected_size = int(data["bytes"])
    expected_sha = str(data["sha256"]).lower()
    if not path.is_file():
        if not download:
            raise FileNotFoundError(f"Offline LIBERO dataset is missing: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(str(data["url"])) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(path)
    if path.stat().st_size != expected_size:
        raise RuntimeError(
            f"LIBERO dataset size mismatch: {path.stat().st_size} != {expected_size}"
        )
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha:
        raise RuntimeError(f"LIBERO dataset SHA-256 mismatch: {observed_sha}")
    return path


def _maybe_limit(dataset: Dataset, maximum: int | None) -> Dataset:
    if maximum is None or len(dataset) <= maximum:
        return dataset
    return Subset(dataset, range(maximum))


def _train_guide(
    model: LiberoVLAPolicy,
    dataset: Dataset,
    config: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    guide = config["guide"]
    loader = DataLoader(
        dataset,
        batch_size=int(guide["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(guide["learning_rate"]),
        weight_decay=float(guide["weight_decay"]),
    )
    iterator = iter(loader)
    metrics: list[dict[str, Any]] = []
    model.train()
    for step in range(int(guide["steps"])):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = libero_policy_loss(model, batch)
        loss.backward()
        optimizer.step()
        metrics.append({"stage": "guide", "step": step, "loss": float(loss.detach().cpu())})
    return metrics


@torch.no_grad()
def _encode_dataset(
    model: LiberoVLAPolicy, dataset: Dataset, *, batch_size: int, device: torch.device
) -> Tensor:
    model.eval()
    encoded: list[Tensor] = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        batch = move_batch_to_device(batch, device)
        representation = model.encode(
            batch["image"], batch["instruction"], batch["proprio"]
        )
        encoded.append(torch.cat([representation, batch["action"]], dim=1).cpu())
    return torch.cat(encoded)


def _distill(
    real_joint: Tensor,
    weights: Tensor | None,
    config: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, dict[str, Any], list[dict[str, Any]]]:
    settings = config["distillation"]
    set_seed(
        seed,
        deterministic=bool(config.get("runtime", {}).get("deterministic", False)),
        deterministic_warn_only=bool(
            config.get("runtime", {}).get("deterministic_warn_only", False)
        ),
    )
    count = max(1, math.ceil(len(real_joint) * float(settings["coreset_ratio"])))
    distiller = WeightedNCFMDistiller(
        real_joint,
        weights,
        synthetic_count=count,
        num_frequencies=int(settings["num_frequencies"]),
        hidden_dim=int(settings["hidden_dim"]),
        generator_lr=float(settings["generator_lr"]),
        discriminator_lr=float(settings["discriminator_lr"]),
        standardize_features=True,
        device=device,
    )
    trace = distiller.run(
        int(settings["steps"]),
        discriminator_steps=int(settings["discriminator_steps"]),
    )
    convergence = summarize_convergence(
        trace, window_ratio=float(settings["convergence_window_ratio"])
    )
    return distiller.coreset(raw_space=True), asdict(convergence), trace_as_dicts(trace)


def _cpu_state(model: torch.nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def run_libero_seed(
    config: dict[str, Any],
    *,
    seed: int,
    output_root: Path,
    device: torch.device,
    download: bool,
    max_samples: int | None,
) -> dict[str, Any]:
    runtime = config.get("runtime", {})
    set_seed(
        seed,
        deterministic=bool(runtime.get("deterministic", False)),
        deterministic_warn_only=bool(runtime.get("deterministic_warn_only", False)),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    seed_directory = output_root / f"seed_{seed}"
    seed_directory.mkdir(parents=True, exist_ok=True)

    data = config["data"]
    dataset_path = _ensure_dataset(data, download=download)
    metadata = inspect_libero_hdf5(dataset_path)
    split_config = data["demo_split"]
    split = split_demo_keys(
        metadata["demo_keys"],
        train_count=int(split_config["train"]),
        reference_count=int(split_config["influence_reference"]),
        selection_count=int(split_config["selection_validation"]),
        test_count=int(split_config["final_test"]),
        seed=seed,
    )
    dataset_arguments = {
        "instruction": str(data["instruction"]),
        "instruction_length": int(data["instruction_length"]),
        "transition_stride": int(data["transition_stride"]),
    }
    train_dataset = _maybe_limit(
        LiberoHDF5Dataset(dataset_path, split.train, **dataset_arguments), max_samples
    )
    reference_dataset = LiberoHDF5Dataset(
        dataset_path, split.influence_reference, **dataset_arguments
    )
    selection_dataset = LiberoHDF5Dataset(
        dataset_path, split.selection_validation, **dataset_arguments
    )
    test_dataset = LiberoHDF5Dataset(dataset_path, split.final_test, **dataset_arguments)

    policy_arguments = {
        "representation_dim": int(config["guide"]["representation_dim"]),
        "visual_dim": int(config["guide"]["visual_dim"]),
        "language_dim": int(config["guide"]["language_dim"]),
        "proprio_dim": int(config["guide"]["proprio_dim"]),
        "action_dim": 7,
    }
    guide = LiberoVLAPolicy(**policy_arguments).to(device)
    metrics: list[dict[str, Any]] = _train_guide(
        guide, train_dataset, config, seed=seed, device=device
    )
    guide.eval()
    guide.freeze_encoders_for_influence()

    influence = config["influence"]
    reference_loader = DataLoader(
        reference_dataset,
        batch_size=int(influence["reference_batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    hessian_loader = DataLoader(
        train_dataset,
        batch_size=int(influence["hessian_batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    sample_batches = single_sample_batches(train_dataset)
    engine = InfluenceEngine(
        guide,
        libero_policy_loss,
        sign=str(influence["sign"]),
        device=device,
    )
    influence_result = engine.score_samples(
        sample_batches,
        reference_loader,
        hessian_loader,
        lissa_depth=int(influence["lissa_depth"]),
        lissa_damping=float(influence["lissa_damping"]),
        lissa_scale=float(influence["lissa_scale"]),
        lissa_repeats=int(influence["lissa_repeats"]),
    )
    verifier = ContrastiveVerifier(
        guide,
        libero_policy_loss,
        beta=float(influence["beta"]),
        elite_ratio=float(influence["elite_ratio"]),
        ranking=str(influence["ranking"]),
        normalization=str(influence["normalization"]),
        epsilon=float(influence["epsilon"]),
    )
    verification = verifier.verify(
        influence_result.base_scores.to(device),
        sample_batches,
        influence_result.reference_gradient,
    )

    representation_dim = int(config["guide"]["representation_dim"])
    encode_batch_size = int(config["guide"]["encode_batch_size"])
    real_joint = _encode_dataset(
        guide, train_dataset, batch_size=encode_batch_size, device=device
    )
    selection_joint = _encode_dataset(
        guide, selection_dataset, batch_size=encode_batch_size, device=device
    )
    test_joint = _encode_dataset(
        guide, test_dataset, batch_size=encode_batch_size, device=device
    )
    count = max(
        1, math.ceil(len(real_joint) * float(config["distillation"]["coreset_ratio"]))
    )
    random_indices = torch.randperm(
        len(real_joint), generator=torch.Generator().manual_seed(seed + 3)
    )[:count]
    random_joint = real_joint[random_indices]
    influence_only_joint = real_joint[verification.elite_indices.cpu()]
    uniform_joint, uniform_convergence, uniform_trace = _distill(
        real_joint, None, config, seed=seed + 10_000, device=device
    )
    ft_joint, ft_convergence, ft_trace = _distill(
        real_joint,
        verification.normalized_weights.detach().cpu(),
        config,
        seed=seed + 10_000,
        device=device,
    )
    for name, trace in (("uniform_ncfm", uniform_trace), ("ft_ncfm", ft_trace)):
        metrics.extend({"stage": "distillation", "variant": name, **item} for item in trace)

    variant_joint = {
        "full": real_joint,
        "random_5pct": random_joint,
        "influence_only_5pct": influence_only_joint,
        "uniform_ncfm": uniform_joint,
        "ft_ncfm": ft_joint,
    }
    downstream = config["downstream"]
    variant_summaries: dict[str, Any] = {}
    action_heads: dict[str, torch.nn.Module] = {}
    for name, joint in variant_joint.items():
        head, summary, trace = train_action_regressor(
            joint,
            selection_joint,
            representation_dim=representation_dim,
            hidden_dim=int(downstream["hidden_dim"]),
            steps=int(downstream["steps"]),
            batch_size=int(downstream["batch_size"]),
            learning_rate=float(downstream["learning_rate"]),
            weight_decay=float(downstream["weight_decay"]),
            seed=seed + 20_000,
            device=device,
            test_joint_features=test_joint,
        )
        variant_summaries[name] = {"downstream": asdict(summary)}
        action_heads[name] = head
        metrics.extend({"stage": "downstream", "variant": name, **item} for item in trace)
    variant_summaries["uniform_ncfm"]["convergence"] = uniform_convergence
    variant_summaries["ft_ncfm"]["convergence"] = ft_convergence

    checkpoint_path = seed_directory / "coreset.pt"
    torch.save(
        {
            "schema": "libero_latent_coreset_policy_v1",
            "policy_arguments": policy_arguments,
            "guide_state_dict": _cpu_state(guide),
            "action_heads": {
                name: _cpu_state(head) for name, head in action_heads.items()
            },
            "representation_dim": representation_dim,
            "action_dim": 7,
            "downstream_hidden_dim": int(downstream["hidden_dim"]),
            "instruction_length": int(data["instruction_length"]),
            "task": {
                "suite": data["suite"],
                "task_name": data["task_name"],
                "instruction": data["instruction"],
            },
            "variants": variant_joint,
        },
        checkpoint_path,
    )

    rollout_config = config.get("rollout", {})
    rollout = None
    if bool(rollout_config.get("enabled", False)):
        rollout = evaluate_libero_checkpoint(
            checkpoint_path,
            libero_root=str(rollout_config["libero_root"]),
            suite=str(data["suite"]),
            task_name=str(data["task_name"]),
            instruction=str(data["instruction"]),
            variants=list(variant_joint),
            episodes=int(rollout_config["episodes"]),
            max_steps=int(rollout_config["max_steps"]),
            image_size=int(data["image_size"]),
            seed=seed,
            device=device,
        )

    summary = {
        "schema": "libero_single_task_distillation",
        "seed": seed,
        "task": {
            "suite": data["suite"],
            "task_name": data["task_name"],
            "instruction": data["instruction"],
        },
        "dataset_sha256": str(data["sha256"]),
        "demo_split": split.as_dict(),
        "train_transitions": len(train_dataset),
        "influence_reference_transitions": len(reference_dataset),
        "selection_validation_transitions": len(selection_dataset),
        "final_test_transitions": len(test_dataset),
        "synthetic_transitions": count,
        "influence_protocol": {
            "sign": influence["sign"],
            "ranking": influence["ranking"],
            "normalization": influence["normalization"],
            "contrastive_implementation": "deterministic visual-only tensor perturbations",
        },
        "variants": variant_summaries,
        "rollout": rollout,
        "elapsed_seconds": time.perf_counter() - start,
    }
    torch.save(
        {
            "sample_ids": influence_result.sample_ids,
            "base_scores": influence_result.base_scores,
            "elite_indices": verification.elite_indices.cpu(),
            "raw_refined_weights": verification.raw_refined_weights.cpu(),
            "normalized_weights": verification.normalized_weights.cpu(),
        },
        seed_directory / "influence.pt",
    )
    write_jsonl(seed_directory / "metrics.jsonl", metrics)
    write_json(seed_directory / "summary.json", summary)
    manifest = {
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "git_sha": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "upstream_sha": git_output("rev-parse", "upstream/main"),
        "libero_sha": str(data["libero_sha"]),
        "dataset_sha256": str(data["sha256"]),
        "config": config,
        "config_sha256": config_hash(config),
        "seed": seed,
        "device": str(device),
        "deterministic_algorithms": bool(runtime.get("deterministic", False)),
        "deterministic_warn_only": bool(runtime.get("deterministic_warn_only", False)),
        "runtime_overrides": {"download": download, "max_samples": max_samples},
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cuda": cuda_metadata(device),
        "execution": {
            "provider": os.getenv("FT_NCFM_EXECUTION_PROVIDER", "local"),
            "runtime_id": os.getenv("FT_NCFM_RUNTIME_ID"),
            "entry_module": os.getenv("FT_NCFM_ENTRY_MODULE", "ft_ncfm.libero_experiment"),
        },
        "packages": {
            "h5py": package_version("h5py"),
            "numpy": package_version("numpy"),
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
            "libero": package_version("libero"),
            "robosuite": package_version("robosuite"),
        },
    }
    write_json(seed_directory / "manifest.json", manifest)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ft_ncfm/libero_spatial_task0.yaml")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
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
        run_libero_seed(
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
