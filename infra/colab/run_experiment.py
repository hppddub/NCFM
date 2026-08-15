"""Run FT-NCFM seeds on Colab with per-seed Drive persistence and resume support."""

from __future__ import annotations

import argparse
import json
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

REQUIRED_SEED_FILES = {
    "coreset.pt",
    "influence.pt",
    "manifest.json",
    "metrics.jsonl",
    "summary.json",
}


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


def is_complete_seed(seed_directory: Path) -> bool:
    if not seed_directory.is_dir():
        return False
    return REQUIRED_SEED_FILES.issubset(path.name for path in seed_directory.iterdir())


def persist_seed(scratch_seed: Path, persistent_seed: Path) -> None:
    if not is_complete_seed(scratch_seed):
        missing = sorted(
            REQUIRED_SEED_FILES.difference(path.name for path in scratch_seed.iterdir())
        )
        raise RuntimeError(f"Seed output is incomplete; missing {missing}")

    partial = persistent_seed.with_name(persistent_seed.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    shutil.copytree(scratch_seed, partial)
    if persistent_seed.exists():
        shutil.rmtree(persistent_seed)
    partial.replace(persistent_seed)


def update_aggregate(persistent_output: Path, seeds: list[int]) -> None:
    summaries = []
    for seed in seeds:
        seed_directory = persistent_output / f"seed_{seed}"
        if is_complete_seed(seed_directory):
            summaries.append(
                json.loads((seed_directory / "summary.json").read_text(encoding="utf-8"))
            )
    write_json(persistent_output / "aggregate_summary.json", {"runs": summaries})


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

    seeds = args.seeds or load_declared_seeds(config_path)
    session_manifest["seeds"] = seeds
    completed: list[int] = []
    skipped: list[int] = []
    environment = os.environ.copy()
    environment["FT_NCFM_EXECUTION_PROVIDER"] = "colab"
    environment["FT_NCFM_RUNTIME_ID"] = session_id

    try:
        for seed in seeds:
            persistent_seed = persistent_output / f"seed_{seed}"
            if is_complete_seed(persistent_seed):
                print(f"Seed {seed} is already complete in Drive; skipping.")
                skipped.append(seed)
                continue

            scratch_seed = scratch_output / f"seed_{seed}"
            if scratch_seed.exists():
                shutil.rmtree(scratch_seed)

            command = [
                sys.executable,
                "-m",
                "ft_ncfm.experiment",
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

            persist_seed(scratch_seed, persistent_seed)
            completed.append(seed)
            update_aggregate(persistent_output, seeds)
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
    update_aggregate(persistent_output, seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
