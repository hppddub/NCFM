"""Run FT-NCFM seeds on Colab with per-seed Drive persistence and resume support."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from ft_ncfm.config import config_hash, load_config

SEED_PAYLOAD_FILES = {
    "coreset.pt",
    "influence.pt",
    "manifest.json",
    "metrics.jsonl",
    "summary.json",
}
CHECKSUM_FILE = "checksums.json"
REQUIRED_SEED_FILES = SEED_PAYLOAD_FILES | {CHECKSUM_FILE}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_output(repository: Path, *args: str) -> str | None:
    return command_output(["git", "-C", str(repository), *args])


def gpu_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "nvidia_smi": command_output(["nvidia-smi"]),
    }
    if not torch.cuda.is_available():
        return metadata

    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    metadata.update(
        {
            "device_count": torch.cuda.device_count(),
            "device_index": index,
            "device_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "total_memory_gib": round(properties.total_memory / 1024**3, 2),
        }
    )
    return metadata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    value = _read_json_value(path.read_text(encoding="utf-8"), path.name)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _validate_finite_numbers(value: Any, source: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{source} contains non-finite numeric values")
    if isinstance(value, dict):
        for child in value.values():
            _validate_finite_numbers(child, source)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_finite_numbers(child, source)


def _read_json_value(text: str, source: str) -> Any:
    value = json.loads(text, parse_constant=_reject_json_constant)
    _validate_finite_numbers(value, source)
    return value


def _validate_finite_tensors(value: Any, path: Path) -> None:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(
            value
        ).all():
            raise ValueError(f"{path.name} contains non-finite tensor values")
        return
    if isinstance(value, dict):
        for child in value.values():
            _validate_finite_tensors(child, path)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_finite_tensors(child, path)


def validate_seed_payload(
    seed_directory: Path, expected_seed: int | None = None
) -> dict[str, Any]:
    if not seed_directory.is_dir():
        raise ValueError(f"Seed directory does not exist: {seed_directory}")
    present = {path.name for path in seed_directory.iterdir()}
    missing = sorted(SEED_PAYLOAD_FILES.difference(present))
    if missing:
        raise ValueError(f"missing required payload files {missing}")

    manifest = _read_json_object(seed_directory / "manifest.json")
    summary = _read_json_object(seed_directory / "summary.json")
    manifest_seed = int(manifest["seed"])
    summary_seed = int(summary["seed"])
    if manifest_seed != summary_seed:
        raise ValueError(
            f"manifest seed {manifest_seed} does not match summary seed {summary_seed}"
        )
    if expected_seed is not None and manifest_seed != expected_seed:
        raise ValueError(
            f"artifact seed {manifest_seed} does not match directory seed {expected_seed}"
        )

    metric_records = 0
    with (seed_directory / "metrics.jsonl").open(encoding="utf-8") as metrics:
        for line_number, line in enumerate(metrics, start=1):
            if not line.strip():
                continue
            record = _read_json_value(line, f"metrics.jsonl line {line_number}")
            if not isinstance(record, dict):
                raise ValueError(f"metrics.jsonl line {line_number} is not an object")
            metric_records += 1
    if metric_records == 0:
        raise ValueError("metrics.jsonl contains no records")

    for filename in ("influence.pt", "coreset.pt"):
        artifact = torch.load(
            seed_directory / filename,
            map_location="cpu",
            weights_only=True,
        )
        _validate_finite_tensors(artifact, seed_directory / filename)

    return {"manifest": manifest, "summary": summary}


def write_seed_checksums(seed_directory: Path) -> None:
    write_json(
        seed_directory / CHECKSUM_FILE,
        {
            "algorithm": "sha256",
            "files": {
                filename: sha256_file(seed_directory / filename)
                for filename in sorted(SEED_PAYLOAD_FILES)
            },
        },
    )


def _experiment_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    identity = {
        "config_sha256": manifest.get("config_sha256"),
        "git_sha": manifest.get("git_sha"),
        "upstream_sha": manifest.get("upstream_sha"),
        "provider": manifest.get("execution", {}).get("provider"),
        "experiment": manifest.get("config", {}).get("experiment", {}).get("name"),
        "runtime_overrides": manifest.get("runtime_overrides"),
        "git_dirty": manifest.get("git_dirty"),
        "device": manifest.get("device"),
        "gpu": manifest.get("cuda", {}).get("device_name"),
    }
    required = (
        "config_sha256",
        "git_sha",
        "upstream_sha",
        "provider",
        "experiment",
        "runtime_overrides",
        "git_dirty",
        "device",
    )
    missing = [field for field in required if identity[field] is None]
    if missing:
        raise ValueError(f"manifest is missing experiment identity fields {missing}")
    return identity


def _validate_experiment_identity(
    manifest: dict[str, Any], expected_identity: dict[str, Any] | None
) -> None:
    actual_identity = _experiment_identity(manifest)
    if expected_identity is not None and actual_identity != expected_identity:
        raise ValueError(
            f"artifact identity {actual_identity!r} does not match requested run "
            f"{expected_identity!r}"
        )


def validate_complete_seed(
    seed_directory: Path,
    expected_seed: int | None = None,
    expected_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = validate_seed_payload(seed_directory, expected_seed)
    _validate_experiment_identity(metadata["manifest"], expected_identity)
    checksums = _read_json_object(seed_directory / CHECKSUM_FILE)
    if checksums.get("algorithm") != "sha256":
        raise ValueError("checksums.json must declare sha256")
    recorded = checksums.get("files")
    if not isinstance(recorded, dict) or set(recorded) != SEED_PAYLOAD_FILES:
        raise ValueError("checksums.json does not cover the exact payload file set")
    for filename, expected_digest in recorded.items():
        actual_digest = sha256_file(seed_directory / filename)
        if actual_digest != expected_digest:
            raise ValueError(f"checksum mismatch for {filename}")
    return metadata


def is_complete_seed(
    seed_directory: Path,
    expected_seed: int | None = None,
    expected_identity: dict[str, Any] | None = None,
) -> bool:
    try:
        validate_complete_seed(seed_directory, expected_seed, expected_identity)
    except Exception:
        return False
    return True


def persist_seed(
    scratch_seed: Path,
    persistent_seed: Path,
    expected_identity: dict[str, Any] | None = None,
) -> None:
    try:
        expected_seed = int(persistent_seed.name.removeprefix("seed_"))
    except ValueError as error:
        raise RuntimeError(f"Invalid persistent seed directory: {persistent_seed}") from error
    try:
        metadata = validate_seed_payload(scratch_seed, expected_seed)
        _validate_experiment_identity(metadata["manifest"], expected_identity)
    except Exception as error:
        raise RuntimeError(f"Seed output is incomplete or invalid: {error}") from error

    partial = persistent_seed.with_name(persistent_seed.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    shutil.copytree(scratch_seed, partial)
    write_seed_checksums(partial)
    try:
        validate_complete_seed(partial, expected_seed, expected_identity)
    except Exception as error:
        raise RuntimeError(f"Persisted seed failed integrity validation: {error}") from error
    if persistent_seed.exists():
        shutil.rmtree(persistent_seed)
    partial.replace(persistent_seed)


def update_aggregate(
    persistent_output: Path,
    expected_seeds: list[int] | None = None,
    expected_identity: dict[str, Any] | None = None,
) -> None:
    """Aggregate compatible, integrity-checked seeds from every invocation."""
    expected = set(expected_seeds) if expected_seeds is not None else None
    complete_seeds: list[tuple[int, dict[str, Any]]] = []
    for seed_directory in persistent_output.glob("seed_*"):
        try:
            seed = int(seed_directory.name.removeprefix("seed_"))
        except ValueError:
            continue
        if expected is not None and seed not in expected:
            raise RuntimeError(f"Persisted seed {seed} is not declared by the config")
        present = {path.name for path in seed_directory.iterdir()}
        if not REQUIRED_SEED_FILES.issubset(present):
            missing = sorted(REQUIRED_SEED_FILES.difference(present))
            raise RuntimeError(f"Persisted seed {seed} is incomplete; missing {missing}")
        try:
            metadata = validate_complete_seed(
                seed_directory, seed, expected_identity
            )
        except Exception as error:
            raise RuntimeError(f"Persisted seed {seed} failed validation: {error}") from error
        complete_seeds.append((seed, metadata))

    identities = [
        (seed, _experiment_identity(metadata["manifest"]))
        for seed, metadata in complete_seeds
    ]
    if identities:
        reference_seed, reference_identity = identities[0]
        for seed, identity in identities[1:]:
            if identity != reference_identity:
                raise RuntimeError(
                    f"Persisted seed {seed} is incompatible with seed {reference_seed}: "
                    f"{identity!r} != {reference_identity!r}"
                )

    summaries = [
        metadata["summary"] for _, metadata in sorted(complete_seeds, key=lambda item: item[0])
    ]
    write_json(persistent_output / "aggregate_summary.json", {"runs": summaries})


def expected_run_identity(
    repository: Path,
    config: dict[str, Any],
    gpu: dict[str, Any],
    *,
    max_samples: int | None,
    offline: bool,
) -> dict[str, Any]:
    identity = {
        "config_sha256": config_hash(config),
        "git_sha": git_output(repository, "rev-parse", "HEAD"),
        "upstream_sha": git_output(repository, "rev-parse", "upstream/main"),
        "provider": "colab",
        "experiment": config.get("experiment", {}).get("name"),
        "runtime_overrides": {
            "download": not offline,
            "max_samples": max_samples,
        },
        "git_dirty": bool(git_output(repository, "status", "--porcelain")),
        "device": "cuda" if gpu["cuda_available"] else "cpu",
        "gpu": gpu.get("device_name"),
    }
    missing = [
        field
        for field in (
            "config_sha256",
            "git_sha",
            "upstream_sha",
            "provider",
            "experiment",
            "runtime_overrides",
            "git_dirty",
            "device",
        )
        if identity[field] is None
    ]
    if missing:
        raise RuntimeError(f"Cannot establish requested run identity; missing {missing}")
    return identity


def effective_experiment_config(
    declared_config: dict[str, Any], scratch_output: Path
) -> dict[str, Any]:
    """Mirror the child experiment's output-dir override before identity hashing."""
    effective_config = copy.deepcopy(declared_config)
    effective_config["experiment"]["output_dir"] = str(scratch_output)
    return effective_config


def load_declared_seeds(config_path: Path) -> list[int]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return [int(seed) for seed in config["experiment"]["seeds"]]


def named_run_directory(root: Path, run_name: str) -> Path:
    if not run_name or run_name in {".", ".."} or Path(run_name).name != run_name:
        raise ValueError("run-name must be one non-empty directory name")
    return root.expanduser().resolve() / run_name


def stream_command(command: list[str], log_path: Path, environment: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Experiment YAML path")
    parser.add_argument(
        "--persistent-root",
        required=True,
        help="Mounted Drive directory that will retain completed seed artifacts",
    )
    parser.add_argument(
        "--scratch-root",
        default="/content/ft-ncfm-scratch",
        help="Fast ephemeral directory used while a seed is running",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Unique output name, for example nested-smoke or true-5pct",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--entry-module",
        choices=("ft_ncfm.experiment", "ft_ncfm.ablation", "ft_ncfm.libero_experiment"),
        default="ft_ncfm.experiment",
        help="Experiment module executed once per seed",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-a100", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[2]
    config_path = (repository / args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")

    gpu = gpu_metadata()
    if args.require_gpu and not gpu["cuda_available"]:
        raise RuntimeError("CUDA is unavailable. Select a GPU runtime before continuing.")
    if args.require_a100 and "A100" not in str(gpu.get("device_name", "")).upper():
        raise RuntimeError(
            f"Expected an A100 but Colab allocated {gpu.get('device_name', 'no CUDA device')}."
        )

    persistent_output = named_run_directory(Path(args.persistent_root), args.run_name)
    scratch_output = named_run_directory(Path(args.scratch_root), args.run_name)
    persistent_output.mkdir(parents=True, exist_ok=True)
    scratch_output.mkdir(parents=True, exist_ok=True)
    session_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    session_manifest = {
        "session_id": session_id,
        "started_at": utc_now(),
        "repository": str(repository),
        "git_sha": git_output(repository, "rev-parse", "HEAD"),
        "git_dirty": bool(git_output(repository, "status", "--porcelain")),
        "config": str(config_path.relative_to(repository)),
        "run_name": args.run_name,
        "entry_module": args.entry_module,
        "max_samples": args.max_samples,
        "offline": args.offline,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu,
        "colab": {
            "release_tag": os.getenv("COLAB_RELEASE_TAG"),
            "backend_version": os.getenv("COLAB_BACKEND_VERSION"),
            "runtime_version": os.getenv("COLAB_RUNTIME_VERSION"),
        },
        "status": "preflight" if args.preflight_only else "running",
    }
    session_path = persistent_output / "sessions" / f"session_{session_id}.json"
    write_json(session_path, session_manifest)

    print(json.dumps(session_manifest, indent=2, sort_keys=True))
    if args.preflight_only:
        session_manifest.update({"finished_at": utc_now(), "status": "preflight_passed"})
        write_json(session_path, session_manifest)
        return 0

    declared_config = load_config(config_path)
    config = effective_experiment_config(declared_config, scratch_output)
    declared_seeds = [int(seed) for seed in declared_config["experiment"]["seeds"]]
    seeds = args.seeds or declared_seeds
    undeclared_seeds = sorted(set(seeds).difference(declared_seeds))
    if undeclared_seeds:
        raise ValueError(f"Requested seeds are not declared by the config: {undeclared_seeds}")
    run_identity = expected_run_identity(
        repository,
        config,
        gpu,
        max_samples=args.max_samples,
        offline=args.offline,
    )
    session_manifest["seeds"] = seeds
    session_manifest["expected_run_identity"] = run_identity
    completed: list[int] = []
    skipped: list[int] = []
    environment = os.environ.copy()
    environment["FT_NCFM_EXECUTION_PROVIDER"] = "colab"
    environment["FT_NCFM_RUNTIME_ID"] = session_id
    environment["FT_NCFM_ENTRY_MODULE"] = args.entry_module

    try:
        update_aggregate(persistent_output, declared_seeds, run_identity)
        for seed in seeds:
            persistent_seed = persistent_output / f"seed_{seed}"
            if is_complete_seed(persistent_seed, seed, run_identity):
                print(f"Seed {seed} is already complete in Drive; skipping.")
                skipped.append(seed)
                continue

            scratch_seed = scratch_output / f"seed_{seed}"
            if scratch_seed.exists():
                shutil.rmtree(scratch_seed)

            command = [
                sys.executable,
                "-m",
                args.entry_module,
                "--config",
                str(config_path),
                "--seed",
                str(seed),
                "--output-dir",
                str(scratch_output),
            ]
            if args.max_samples is not None:
                command.extend(["--max-samples", str(args.max_samples)])
            if args.offline:
                command.append("--offline")

            print(f"Running seed {seed}: {' '.join(command)}")
            seed_started = time.perf_counter()
            log_path = persistent_output / "logs" / f"seed_{seed}_{session_id}.log"
            exit_code = stream_command(command, log_path, environment)
            if exit_code != 0:
                raise RuntimeError(f"Seed {seed} failed with exit code {exit_code}; see {log_path}")

            persist_seed(scratch_seed, persistent_seed, run_identity)
            completed.append(seed)
            update_aggregate(persistent_output, declared_seeds, run_identity)
            session_manifest.update(
                {
                    "completed_seeds": completed,
                    "skipped_seeds": skipped,
                    "last_seed_elapsed_seconds": time.perf_counter() - seed_started,
                }
            )
            write_json(session_path, session_manifest)
            print(f"Seed {seed} is complete and persisted to {persistent_seed}")
    except BaseException as error:
        session_manifest.update(
            {
                "finished_at": utc_now(),
                "status": "failed",
                "completed_seeds": completed,
                "skipped_seeds": skipped,
                "error": repr(error),
            }
        )
        write_json(session_path, session_manifest)
        raise

    session_manifest.update(
        {
            "finished_at": utc_now(),
            "status": "completed",
            "completed_seeds": completed,
            "skipped_seeds": skipped,
        }
    )
    write_json(session_path, session_manifest)
    update_aggregate(persistent_output, declared_seeds, run_identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
