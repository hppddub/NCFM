#!/usr/bin/env python3
"""Install and pin the minimum official LIBERO simulator stack in Colab."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

DEFAULT_SHA = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
DEFAULT_REPOSITORY = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def prepare_checkout(root: Path, repository: str, sha: str) -> None:
    if not (root / ".git").is_dir():
        run(["git", "clone", "--filter=blob:none", repository, str(root)])
    run(["git", "-C", str(root), "fetch", "--depth", "1", "origin", sha])
    run(["git", "-C", str(root), "checkout", "--detach", sha])
    observed = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != sha:
        raise RuntimeError(f"LIBERO checkout mismatch: {observed} != {sha}")


def install_dependencies(root: Path) -> None:
    # The official requirements pin a Python-3.8-era training stack. The
    # one-task evaluator only imports the simulator subset below.
    packages = [
        "h5py>=3.11,<4",
        "mujoco==2.3.7",
        "robosuite==1.4.0",
        "bddl==1.0.1",
        "gym==0.25.2",
        "cloudpickle==2.2.1",
        "future==1.0.0",
        "termcolor==2.4.0",
        "opencv-python-headless==4.10.0.84",
        "matplotlib==3.8.4",
    ]
    run([sys.executable, "-m", "pip", "install", *packages])
    run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(root)])


def write_config(root: Path, config_root: Path) -> Path:
    config_root.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(root / "libero" / "libero"),
        "bddl_files": str(root / "libero" / "libero" / "bddl_files"),
        "init_states": str(root / "libero" / "libero" / "init_files"),
        "datasets": str(root / "libero" / "datasets"),
        "assets": str(root / "libero" / "libero" / "assets"),
    }
    config_path = config_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/content/LIBERO")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--sha", default=DEFAULT_SHA)
    parser.add_argument("--config-root", default=str(Path.home() / ".libero"))
    parser.add_argument("--skip-install", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root).expanduser().resolve()
    config_root = Path(args.config_root).expanduser().resolve()
    prepare_checkout(root, args.repository, args.sha)
    if not args.skip_install:
        install_dependencies(root)
    config_path = write_config(root, config_root)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    # Subprocess import proves that a fresh process sees the same noninteractive
    # config and simulator packages used by the experiment runner.
    environment = os.environ.copy()
    environment["LIBERO_CONFIG_PATH"] = str(config_root)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, libero, mujoco, robosuite; "
                "from libero.libero.envs import OffScreenRenderEnv; "
                "print(json.dumps({'libero': libero.__file__, "
                "'mujoco': mujoco.__version__, "
                "'robosuite': getattr(robosuite, '__version__', 'unknown')}))"
            ),
        ],
        check=True,
        env=environment,
    )
    print(
        json.dumps(
            {
                "libero_root": str(root),
                "libero_sha": args.sha,
                "config": str(config_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
