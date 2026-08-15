import ast
import json
from pathlib import Path

import pytest

from infra.colab.run_experiment import (
    REQUIRED_SEED_FILES,
    is_complete_seed,
    named_run_directory,
    persist_seed,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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


def test_colab_runner_persists_only_a_complete_seed(tmp_path: Path) -> None:
    scratch_seed = tmp_path / "scratch" / "seed_42"
    persistent_seed = tmp_path / "drive" / "seed_42"
    scratch_seed.mkdir(parents=True)
    for filename in REQUIRED_SEED_FILES:
        (scratch_seed / filename).write_text(filename, encoding="utf-8")

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


@pytest.mark.parametrize("run_name", ["", ".", "..", "../escape", "nested/name"])
def test_colab_runner_rejects_unsafe_run_names(tmp_path: Path, run_name: str) -> None:
    with pytest.raises(ValueError, match="run-name"):
        named_run_directory(tmp_path, run_name)


def test_experiment_manifest_declares_colab_provenance_fields() -> None:
    experiment = (REPOSITORY_ROOT / "ft_ncfm/experiment.py").read_text(encoding="utf-8")

    assert '"provider": os.getenv("FT_NCFM_EXECUTION_PROVIDER", "local")' in experiment
    assert '"release_tag": os.getenv("COLAB_RELEASE_TAG")' in experiment
    assert '"runtime_id": os.getenv("FT_NCFM_RUNTIME_ID")' in experiment
