"""Run the preregistered sign/ranking/normalization Mini-VLA proxy ablation."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import platform
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .config import config_hash, load_config
from .contrastive import ContrastiveVerifier
from .data import load_mnist_minivla_splits
from .distillation import WeightedNCFMDistiller, summarize_convergence, trace_as_dicts
from .downstream import evaluate_action_mse, train_action_regressor
from .experiment import (
    cuda_metadata,
    encode_dataset,
    git_output,
    maybe_limit_dataset,
    package_version,
    policy_loss,
    set_seed,
    single_sample_batches,
    train_guide,
    write_json,
    write_jsonl,
)
from .influence import InfluenceEngine
from .models import MiniVLAPolicy


def condition_name(sign: str, ranking: str, normalization: str) -> str:
    return f"sign={sign}__rank={ranking}__norm={normalization}"


def _distill(
    real_joint_features: Tensor,
    weights: Tensor | None,
    config: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, dict[str, Any], list[dict[str, float | int]]]:
    distillation = config["distillation"]
    set_seed(
        seed,
        deterministic=bool(config.get("runtime", {}).get("deterministic", False)),
        deterministic_warn_only=bool(
            config.get("runtime", {}).get("deterministic_warn_only", False)
        ),
    )
    synthetic_count = max(
        1,
        math.ceil(
            real_joint_features.shape[0] * float(distillation["coreset_ratio"])
        ),
    )
    distiller = WeightedNCFMDistiller(
        real_joint_features,
        weights,
        synthetic_count=synthetic_count,
        num_frequencies=int(distillation["num_frequencies"]),
        hidden_dim=int(distillation["hidden_dim"]),
        generator_lr=float(distillation["generator_lr"]),
        discriminator_lr=float(distillation["discriminator_lr"]),
        standardize_features=True,
        device=device,
    )
    trace = distiller.run(
        int(distillation["steps"]),
        discriminator_steps=int(distillation["discriminator_steps"]),
    )
    convergence = summarize_convergence(
        trace,
        window_ratio=float(distillation["convergence_window_ratio"]),
    )
    return distiller.coreset(raw_space=True), asdict(convergence), trace_as_dicts(trace)


def _downstream(
    train_joint_features: Tensor,
    selection_joint_features: Tensor,
    test_joint_features: Tensor | None,
    config: dict[str, Any],
    *,
    seed: int,
    representation_dim: int,
    device: torch.device,
):
    downstream = config["downstream"]
    return train_action_regressor(
        train_joint_features,
        selection_joint_features,
        representation_dim=representation_dim,
        hidden_dim=int(downstream["hidden_dim"]),
        steps=int(downstream["steps"]),
        batch_size=int(downstream["batch_size"]),
        learning_rate=float(downstream["learning_rate"]),
        weight_decay=float(downstream["weight_decay"]),
        seed=seed,
        device=device,
        test_joint_features=test_joint_features,
    )


def _append_trace(
    metrics: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    *,
    stage: str,
    variant: str,
) -> None:
    metrics.extend({"stage": stage, "variant": variant, **record} for record in trace)


def run_ablation_seed(
    config: dict[str, Any],
    *,
    seed: int,
    output_root: Path,
    device: torch.device,
    download: bool,
    max_samples: int | None,
) -> dict[str, Any]:
    runtime = config.get("runtime", {})
    deterministic = bool(runtime.get("deterministic", False))
    deterministic_warn_only = bool(runtime.get("deterministic_warn_only", False))
    set_seed(
        seed,
        deterministic=deterministic,
        deterministic_warn_only=deterministic_warn_only,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started_at = datetime.now(UTC).isoformat()
    start = time.perf_counter()
    seed_directory = output_root / f"seed_{seed}"
    seed_directory.mkdir(parents=True, exist_ok=True)

    data = config["data"]
    train_dataset, reference_dataset, selection_dataset, test_dataset = (
        load_mnist_minivla_splits(
            str(data["cache_dir"]),
            train_ratio=float(data["train_ratio"]),
            reference_size=int(data["reference_size"]),
            selection_size=int(data["selection_size"]),
            test_size=int(data["test_size"]),
            seed=seed,
            download=download,
        )
    )
    train_dataset = maybe_limit_dataset(train_dataset, max_samples)
    if max_samples is not None:
        reference_dataset = maybe_limit_dataset(reference_dataset, 64)
        selection_dataset = maybe_limit_dataset(selection_dataset, 128)
        test_dataset = maybe_limit_dataset(test_dataset, 128)

    guide = MiniVLAPolicy(hidden_dim=int(config["guide"]["hidden_dim"])).to(device)
    metrics: list[dict[str, Any]] = list(
        train_guide(guide, train_dataset, config, seed=seed, device=device)
    )
    guide.eval()

    reference_batches = torch.utils.data.DataLoader(
        reference_dataset,
        batch_size=int(config["influence"]["reference_batch_size"]),
        shuffle=False,
    )
    hessian_batches = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(config["influence"]["hessian_batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    sample_batches = single_sample_batches(train_dataset)
    influence = config["influence"]
    engine = InfluenceEngine(
        guide,
        policy_loss,
        sign="gradient_alignment",
        device=device,
    )
    alignment_result = engine.score_samples(
        sample_batches,
        reference_batches,
        hessian_batches,
        lissa_depth=int(influence["lissa_depth"]),
        lissa_damping=float(influence["lissa_damping"]),
        lissa_scale=float(influence["lissa_scale"]),
        lissa_repeats=int(influence["lissa_repeats"]),
    )

    representation_dim = int(config["guide"]["hidden_dim"])
    real_joint = encode_dataset(guide, train_dataset, device=device)
    selection_joint = encode_dataset(guide, selection_dataset, device=device)
    test_joint = encode_dataset(guide, test_dataset, device=device)

    baseline_summaries: dict[str, Any] = {}
    baseline_coresets: dict[str, Tensor] = {}
    downstream_seed = seed + 20_000
    full_model, full_summary, full_trace = _downstream(
        real_joint,
        selection_joint,
        test_joint,
        config,
        seed=downstream_seed,
        representation_dim=representation_dim,
        device=device,
    )
    del full_model
    baseline_summaries["full"] = {"downstream": asdict(full_summary)}
    _append_trace(metrics, full_trace, stage="downstream", variant="full")

    coreset_count = max(
        1, math.ceil(real_joint.shape[0] * float(config["distillation"]["coreset_ratio"]))
    )
    random_indices = torch.randperm(
        real_joint.shape[0], generator=torch.Generator().manual_seed(seed + 3)
    )[:coreset_count]
    random_joint = real_joint[random_indices]
    random_model, random_summary, random_trace = _downstream(
        random_joint,
        selection_joint,
        test_joint,
        config,
        seed=downstream_seed,
        representation_dim=representation_dim,
        device=device,
    )
    del random_model
    baseline_coresets["random_5pct"] = random_joint
    baseline_summaries["random_5pct"] = {"downstream": asdict(random_summary)}
    _append_trace(metrics, random_trace, stage="downstream", variant="random_5pct")

    uniform_joint, uniform_convergence, uniform_distillation_trace = _distill(
        real_joint,
        None,
        config,
        seed=seed + 10_000,
        device=device,
    )
    uniform_model, uniform_summary, uniform_downstream_trace = _downstream(
        uniform_joint,
        selection_joint,
        test_joint,
        config,
        seed=downstream_seed,
        representation_dim=representation_dim,
        device=device,
    )
    del uniform_model
    baseline_coresets["uniform_ncfm"] = uniform_joint
    baseline_summaries["uniform_ncfm"] = {
        "convergence": uniform_convergence,
        "downstream": asdict(uniform_summary),
    }
    _append_trace(
        metrics,
        uniform_distillation_trace,
        stage="distillation",
        variant="uniform_ncfm",
    )
    _append_trace(
        metrics,
        uniform_downstream_trace,
        stage="downstream",
        variant="uniform_ncfm",
    )

    ablation = config["ablation"]
    condition_summaries: dict[str, Any] = {}
    condition_weights: dict[str, Tensor] = {}
    condition_elites: dict[str, Tensor] = {}
    condition_coresets: dict[str, Tensor] = {}
    condition_models: dict[str, torch.nn.Module] = {}
    for sign, ranking, normalization in itertools.product(
        ablation["signs"], ablation["rankings"], ablation["normalizations"]
    ):
        name = condition_name(str(sign), str(ranking), str(normalization))
        base_scores = (
            -alignment_result.base_scores
            if sign == "paper"
            else alignment_result.base_scores
        )
        verifier = ContrastiveVerifier(
            guide,
            policy_loss,
            beta=float(influence["beta"]),
            elite_ratio=float(influence["elite_ratio"]),
            ranking=str(ranking),
            normalization=str(normalization),
            epsilon=float(influence["epsilon"]),
        )
        verification = verifier.verify(
            base_scores.to(device),
            sample_batches,
            alignment_result.reference_gradient,
        )
        probabilities = verification.normalized_weights.detach().cpu()
        synthetic_joint, convergence, distillation_trace = _distill(
            real_joint,
            probabilities,
            config,
            seed=seed + 10_000,
            device=device,
        )
        action_model, downstream_summary, downstream_trace = _downstream(
            synthetic_joint,
            selection_joint,
            None,
            config,
            seed=downstream_seed,
            representation_dim=representation_dim,
            device=device,
        )
        condition_summaries[name] = {
            "sign": sign,
            "ranking": ranking,
            "normalization": normalization,
            "elite_samples": int(verification.elite_indices.numel()),
            "convergence": convergence,
            "downstream": asdict(downstream_summary),
        }
        condition_weights[name] = probabilities
        condition_elites[name] = verification.elite_indices.cpu()
        condition_coresets[name] = synthetic_joint
        condition_models[name] = action_model
        _append_trace(metrics, distillation_trace, stage="distillation", variant=name)
        _append_trace(metrics, downstream_trace, stage="downstream", variant=name)

    selected_condition = min(
        condition_summaries,
        key=lambda name: condition_summaries[name]["downstream"]["selection_mse"],
    )
    selected_test_mse = evaluate_action_mse(
        condition_models[selected_condition],
        test_joint,
        representation_dim=representation_dim,
        device=device,
    )
    condition_summaries[selected_condition]["downstream"]["test_mse"] = selected_test_mse

    selected_elites = condition_elites[selected_condition]
    influence_only_joint = real_joint[selected_elites]
    influence_model, influence_summary, influence_trace = _downstream(
        influence_only_joint,
        selection_joint,
        test_joint,
        config,
        seed=downstream_seed,
        representation_dim=representation_dim,
        device=device,
    )
    del influence_model
    baseline_coresets["influence_only_5pct"] = influence_only_joint
    baseline_summaries["influence_only_5pct"] = {
        "selected_from": selected_condition,
        "downstream": asdict(influence_summary),
    }
    _append_trace(
        metrics,
        influence_trace,
        stage="downstream",
        variant="influence_only_5pct",
    )

    elapsed = time.perf_counter() - start
    summary = {
        "schema": "minivla_factorial_ablation",
        "seed": seed,
        "source_train_samples": len(train_dataset),
        "influence_reference_samples": len(reference_dataset),
        "selection_validation_samples": len(selection_dataset),
        "final_test_samples": len(test_dataset),
        "synthetic_samples": coreset_count,
        "conditions": condition_summaries,
        "selected_condition": selected_condition,
        "selection_rule": "minimum isolated selection-validation action MSE",
        "selected_final_test_mse": selected_test_mse,
        "baselines": baseline_summaries,
        "elapsed_seconds": elapsed,
    }
    torch.save(
        {
            "sample_ids": alignment_result.sample_ids,
            "gradient_alignment_scores": alignment_result.base_scores,
            "condition_weights": condition_weights,
            "condition_elites": condition_elites,
        },
        seed_directory / "influence.pt",
    )
    torch.save(
        {
            "schema": "trainable_joint_latent_action",
            "representation_dim": representation_dim,
            "baselines": baseline_coresets,
            "conditions": condition_coresets,
            "selected_condition": selected_condition,
        },
        seed_directory / "coreset.pt",
    )
    write_jsonl(seed_directory / "metrics.jsonl", metrics)
    write_json(seed_directory / "summary.json", summary)
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
            "entry_module": os.getenv("FT_NCFM_ENTRY_MODULE", "ft_ncfm.ablation"),
        },
        "packages": {
            "numpy": package_version("numpy"),
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
        },
    }
    write_json(seed_directory / "manifest.json", manifest)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/ft_ncfm/minivla_factorial_ablation.yaml",
    )
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
        run_ablation_seed(
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
