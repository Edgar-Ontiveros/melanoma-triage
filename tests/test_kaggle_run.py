"""kaggle_run: el metadata del kernel sale de configs/kaggle.yaml y los parámetros del notebook se
sustituyen sin ejecutar nada. No toca la red."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from melanoma.utils.notebook import load_notebook, read_parameters, substitute_parameters

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("kaggle_run", ROOT / "scripts/kaggle_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_substitute_parameters_roundtrip() -> None:
    nb = load_notebook(ROOT / "notebooks/kaggle_train.ipynb")
    before = read_parameters(nb)
    assert before["REPO_SHA"] == "main" and isinstance(before["OVERRIDES"], list)
    after = read_parameters(
        substitute_parameters(
            nb, {"REPO_SHA": "abc1234", "OVERRIDES": ["+experiment=prep_b_divide255"]}
        )
    )
    assert after["REPO_SHA"] == "abc1234"
    assert after["OVERRIDES"] == ["+experiment=prep_b_divide255"]
    assert after["WORK"] == before["WORK"]
    with pytest.raises(KeyError):
        substitute_parameters(nb, {"NO_EXISTE": 1})


def test_kernel_metadata_from_config() -> None:
    kr = _load_script()
    c = kr.cfg()
    prep = kr.kernel_metadata("prepare", c, "kaggle_prepare_512.ipynb")
    assert (
        prep["id"] == f"{c.username}/melanoma-f2-prepare-512"
        and prep["title"] == "melanoma-f2-prepare-512"
    )
    assert prep["enable_gpu"] == "false" and prep["enable_internet"] == "true"
    assert prep["competition_sources"] == [c.competition]
    assert prep["dataset_sources"] == [c.datasets.splits]
    train = kr.kernel_metadata("b1-s0", c, "kaggle_train.ipynb")
    assert train["id"].endswith("/melanoma-f2-train-b1-s0")
    assert train["enable_gpu"] == "true" and train["machine_shape"] == c.gpu_machine_shape
    assert set(train["dataset_sources"]) == {c.datasets.splits, c.datasets.images512}
    assert train["competition_sources"] == []
    for key in (
        "code_file",
        "language",
        "kernel_type",
        "is_private",
        "kernel_sources",
        "model_sources",
    ):
        assert key in train


def test_every_run_stages_a_valid_kernel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kr = _load_script()
    monkeypatch.setattr(kr, "ROOT", ROOT)
    c = kr.cfg()
    seen = set()
    for name in list(c.runs) + ["prepare"]:
        folder = kr.stage_kernel(name, c, sha="deadbeef", extra={})
        meta = json.loads((folder / "kernel-metadata.json").read_text())
        params = json.loads((folder / "params.json").read_text())
        assert params["REPO_SHA"] == "deadbeef"
        assert (folder / meta["code_file"]).exists()
        assert meta["id"] not in seen, f"slug repetido: {meta['id']}"
        seen.add(meta["id"])
        nb = load_notebook(folder / meta["code_file"])
        assert read_parameters(nb)["REPO_SHA"] == "deadbeef"
        if name in c.runs:
            assert read_parameters(nb)["RUN_TAG"] == name


def test_offline_wandb_dirs_found(tmp_path: Path) -> None:
    kr = _load_script()
    (tmp_path / "runs/x/wandb/offline-run-20260917_1-abc").mkdir(parents=True)
    (tmp_path / "runs/x/wandb/run-20260917_2-def").mkdir(parents=True)
    found = kr.offline_wandb_dirs(tmp_path)
    assert [p.name for p in found] == ["offline-run-20260917_1-abc"]


def test_network_errors_are_not_terminal() -> None:
    kr = _load_script()
    assert kr.is_terminal("COMPLETE")
    assert kr.is_terminal("ERROR (Failure message)")
    assert kr.is_terminal("CANCELACKNOWLEDGED")
    assert not kr.is_terminal("RUNNING")
    assert not kr.is_terminal("desconocido: HTTPSConnectionPool ... NameResolutionError(...)")
