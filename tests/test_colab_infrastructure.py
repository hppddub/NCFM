import ast
import json
from pathlib import Path

import pytest
import torch

from ft_ncfm.config import config_hash
from infra.colab.run_experiment import (
    REQUIRED_SEED_FILES,
    effective_experiment_config,
    expected_run_identity,
    is_complete_seed,
    named_run_directory,
    persist_seed,
    update_aggregate,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def write_seed_payload(
    seed_directory: Path,
    seed: int,
    *,
    config_sha256: str = "config-sha",
) -> None:
    seed_directory.mkdir(parents=True)
    manifest = {
        "seed": seed,
        "config_sha256": config_sha256,
        "git_sha": "git-sha",
        "git_dirty": False,
        "upstream_sha": "upstream-sha",
        "execution": {"provider": "colab"},
        "config": {"experiment": {"name": "unit-test"}},
        "runtime_overrides": {"download": True, "max_samples": None},
        "device": "cuda",
        "cuda": {"device_name": "NVIDIA A100-SXM4-80GB"},
    }
    (seed_directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (seed_directory / "summary.json").write_text(
        json.dumps({"seed": seed}), encoding="utf-8"
    )
    (seed_directory / "metrics.jsonl").write_text(
        json.dumps({"step": 0, "loss": 1.0}) + "\n", encoding="utf-8"
    )
    torch.save({"scores": torch.tensor([1.0])}, seed_directory / "influence.pt")
    torch.save({"synthetic": torch.tensor([[1.0]])}, seed_directory / "coreset.pt")


def test_colab_notebook_is_clean_and_expensive_runs_are_opt_in() -> None:
    notebook_path = REPOSITORY_ROOT / "notebooks/FT_NCFM_Colab_A100.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    assert "REQUIRE_A100 = True" in sources
    assert "PIN_PAPER_TORCH = True" in sources
    assert "RUN_TRUE_5PCT_SEED_42 = False" in sources
    assert "RUN_REMAINING_TRUE_5PCT_SEEDS = False" in sources
    assert "--preflight-only" in sources
    assert "--max-samples\", \"1000" in sources


def test_colab_runner_exposes_proxy_and_ablation_entry_modules() -> None:
    runner = (REPOSITORY_ROOT / "infra/colab/run_experiment.py").read_text(
        encoding="utf-8"
    )
    assert '"ft_ncfm.experiment"' in runner
    assert '"ft_ncfm.ablation"' in runner
    assert '"ft_ncfm.libero_experiment"' in runner


def test_colab_runner_persists_only_a_complete_seed(tmp_path: Path) -> None:
    scratch_seed = tmp_path / "scratch" / "seed_42"
    persistent_seed = tmp_path / "drive" / "seed_42"
    write_seed_payload(scratch_seed, 42)

    persist_seed(scratch_seed, persistent_seed)

    assert is_complete_seed(persistent_seed)
    assert not persistent_seed.with_name("seed_42.partial").exists()
    assert {path.name for path in persistent_seed.iterdir()} == REQUIRED_SEED_FILES


def test_colab_runner_rejects_incomplete_seed(tmp_path: Path) -> None:
    scratch_seed = tmp_path / "scratch" / "seed_42"
    scratch_seed.mkdir(parents=True)
    (scratch_seed / "summary.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="incomplete"):
        persist_seed(scratch_seed, tmp_path / "drive" / "seed_42")


def test_colab_runner_aggregates_complete_seeds_across_invocations(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    for seed in (42, 123, 1024):
        scratch_seed = tmp_path / "scratch" / f"seed_{seed}"
        write_seed_payload(scratch_seed, seed)
        persist_seed(scratch_seed, output / f"seed_{seed}")

    partial = output / "seed_999.partial"
    partial.mkdir()
    (partial / "summary.json").write_text('{"seed": 999}', encoding="utf-8")

    update_aggregate(output, [42, 123, 1024, 999])

    aggregate = json.loads((output / "aggregate_summary.json").read_text())
    assert [run["seed"] for run in aggregate["runs"]] == [42, 123, 1024]


def test_colab_runner_rejects_incomplete_named_seed(tmp_path: Path) -> None:
    output = tmp_path / "run"
    incomplete = output / "seed_42"
    incomplete.mkdir(parents=True)
    (incomplete / "summary.json").write_text('{"seed": 42}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="incomplete"):
        update_aggregate(output, [42])


def test_colab_runner_rejects_corrupted_complete_seed(tmp_path: Path) -> None:
    scratch_seed = tmp_path / "scratch" / "seed_42"
    persistent_seed = tmp_path / "run" / "seed_42"
    write_seed_payload(scratch_seed, 42)
    persist_seed(scratch_seed, persistent_seed)
    (persistent_seed / "summary.json").write_text('{"seed": 123}', encoding="utf-8")

    assert not is_complete_seed(persistent_seed, 42)
    with pytest.raises(RuntimeError, match="failed validation"):
        update_aggregate(tmp_path / "run", [42])


@pytest.mark.parametrize(
    ("filename", "contents"),
    [
        ("summary.json", '{"seed": 42, "loss": NaN}'),
        ("metrics.jsonl", '{"step": 0, "loss": Infinity}\n'),
    ],
)
def test_colab_runner_rejects_nonfinite_json_values(
    tmp_path: Path, filename: str, contents: str
) -> None:
    scratch_seed = tmp_path / "scratch" / "seed_42"
    write_seed_payload(scratch_seed, 42)
    (scratch_seed / filename).write_text(contents, encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-finite"):
        persist_seed(scratch_seed, tmp_path / "run" / "seed_42")


def test_colab_runner_rejects_incompatible_seed_manifests(tmp_path: Path) -> None:
    output = tmp_path / "run"
    for seed, config_sha in ((42, "config-a"), (123, "config-b")):
        scratch_seed = tmp_path / "scratch" / f"seed_{seed}"
        write_seed_payload(scratch_seed, seed, config_sha256=config_sha)
        persist_seed(scratch_seed, output / f"seed_{seed}")

    with pytest.raises(RuntimeError, match="incompatible"):
        update_aggregate(output, [42, 123])


def test_colab_runner_rejects_stored_seeds_from_another_requested_run(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    for seed in (42, 123):
        scratch_seed = tmp_path / "scratch" / f"seed_{seed}"
        write_seed_payload(scratch_seed, seed, config_sha256="old-config")
        persist_seed(scratch_seed, output / f"seed_{seed}")

    expected_identity = {
        "config_sha256": "new-config",
        "git_sha": "git-sha",
        "git_dirty": False,
        "upstream_sha": "upstream-sha",
        "provider": "colab",
        "experiment": "unit-test",
        "runtime_overrides": {"download": True, "max_samples": None},
        "device": "cuda",
        "gpu": "NVIDIA A100-SXM4-80GB",
    }
    with pytest.raises(RuntimeError, match="requested run"):
        update_aggregate(output, [42, 123], expected_identity)


def test_colab_runner_compares_all_runtime_overrides(tmp_path: Path) -> None:
    output = tmp_path / "run"
    scratch_seed = tmp_path / "scratch" / "seed_42"
    write_seed_payload(scratch_seed, 42)
    persist_seed(scratch_seed, output / "seed_42")

    expected_identity = {
        "config_sha256": "config-sha",
        "git_sha": "git-sha",
        "git_dirty": False,
        "upstream_sha": "upstream-sha",
        "provider": "colab",
        "experiment": "unit-test",
        "runtime_overrides": {"download": False, "max_samples": None},
        "device": "cuda",
        "gpu": "NVIDIA A100-SXM4-80GB",
    }
    with pytest.raises(RuntimeError, match="requested run"):
        update_aggregate(output, [42], expected_identity)


def test_colab_runner_hashes_the_child_output_override(tmp_path: Path) -> None:
    declared = {
        "experiment": {
            "name": "unit-test",
            "output_dir": "artifacts/from-yaml",
            "seeds": [42],
        }
    }
    scratch_output = (tmp_path / "scratch" / "run").resolve()

    effective = effective_experiment_config(declared, scratch_output)
    expected_identity = expected_run_identity(
        REPOSITORY_ROOT,
        effective,
        {"cuda_available": False},
        max_samples=1000,
        offline=True,
    )

    assert declared["experiment"]["output_dir"] == "artifacts/from-yaml"
    assert effective["experiment"]["output_dir"] == str(scratch_output)
    assert config_hash(effective) != config_hash(declared)
    assert expected_identity["config_sha256"] == config_hash(effective)
    assert expected_identity["runtime_overrides"] == {
        "download": False,
        "max_samples": 1000,
    }


@pytest.mark.parametrize("run_name", ["", ".", "..", "../escape", "nested/name"])
def test_colab_runner_rejects_unsafe_run_names(tmp_path: Path, run_name: str) -> None:
    with pytest.raises(ValueError, match="run-name"):
        named_run_directory(tmp_path, run_name)


def test_experiment_manifest_declares_colab_provenance_fields() -> None:
    experiment = (REPOSITORY_ROOT / "ft_ncfm/experiment.py").read_text(encoding="utf-8")

    assert '"provider": os.getenv("FT_NCFM_EXECUTION_PROVIDER", "local")' in experiment
    assert '"release_tag": os.getenv("COLAB_RELEASE_TAG")' in experiment
    assert '"runtime_id": os.getenv("FT_NCFM_RUNTIME_ID")' in experiment
