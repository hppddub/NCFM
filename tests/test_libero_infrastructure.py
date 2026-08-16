from pathlib import Path

import yaml

from infra.colab.bootstrap_libero import DEFAULT_SHA, build_parser, write_config


def test_libero_bootstrap_pins_official_commit() -> None:
    args = build_parser().parse_args([])
    assert args.sha == DEFAULT_SHA == "8f1084e3132a39270c3a13ebe37270a43ece2a01"


def test_libero_bootstrap_writes_noninteractive_paths(tmp_path: Path) -> None:
    root = tmp_path / "LIBERO"
    config_path = write_config(root, tmp_path / ".libero")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["bddl_files"] == str(root / "libero" / "libero" / "bddl_files")
    assert config["init_states"] == str(root / "libero" / "libero" / "init_files")
    assert config["datasets"] == str(root / "libero" / "datasets")
