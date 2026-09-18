"""Los splits son disjuntos por paciente, cubren el manifiesto y sus hashes están en el README."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import DictConfig

from melanoma.data.manifest import read_manifest
from melanoma.data.splits import patients_in_two_splits


def _load(root_dir: Path, cfg: DictConfig) -> tuple[pd.DataFrame, pd.Series, dict[str, Path]]:
    manifest = root_dir / cfg.data.manifest_path
    splits_dir = root_dir / cfg.data.splits_dir
    files = {name: splits_dir / f"{name}.txt" for name in cfg.data.split.names}
    if not manifest.exists() or not all(p.exists() for p in files.values()):
        pytest.skip("faltan el manifiesto o los archivos de split")
    m = read_manifest(manifest)
    parts = []
    for name, path in files.items():
        ids = path.read_text(encoding="utf-8").split()
        assert ids == sorted(ids), f"{path.name} no está ordenado"
        assert len(ids) == len(set(ids)), f"{path.name} tiene repetidos"
        parts.append(pd.Series(name, index=ids, name="split"))
    return m, pd.concat(parts), files


def test_splits_disjoint_by_patient(root_dir: Path, prepare_cfg: DictConfig) -> None:
    m, split, _ = _load(root_dir, prepare_cfg)
    assert split.index.is_unique, "una imagen aparece en dos splits"
    assert patients_in_two_splits(m, split) == []


def test_splits_cover_manifest(root_dir: Path, prepare_cfg: DictConfig) -> None:
    m, split, _ = _load(root_dir, prepare_cfg)
    assert set(split.index) == set(m["image_id"])
    assert len(split) == len(m)


def test_split_hashes_match_readme(root_dir: Path, prepare_cfg: DictConfig) -> None:
    _, _, files = _load(root_dir, prepare_cfg)
    readme = (root_dir / "README.md").read_text(encoding="utf-8")
    found = re.findall(r"^([0-9a-f]{64})\s+(\w+\.txt)\s*$", readme, flags=re.MULTILINE)
    listed = {name: digest for digest, name in found}
    assert files, "no hay archivos de split que verificar"
    for path in files.values():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path.name in listed, f"el README no lista el SHA256 de {path.name}"
        assert listed[path.name] == actual, f"SHA256 de {path.name} distinto al del README"


def test_test_set_access_log_within_budget(root_dir: Path, prepare_cfg: DictConfig) -> None:
    """Presupuesto declarado: dos accesos al conjunto de prueba en todo el proyecto. Cada
    línea lleva timestamp, commit y hash del protocolo (F4.4). Estuvo vacío hasta el
    2026-09-18; el primer acceso fue la evaluación clínica de F4."""
    log = root_dir / prepare_cfg.data.test_access_log
    assert log.exists(), "falta logs/test_set_access.log"
    lines = [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) <= 2, f"presupuesto de accesos al test excedido: {len(lines)} > 2"
    for line in lines:
        fields = dict(f.split("=", 1) for f in line.split("\t")[1:])
        assert re.match(r"^\d{4}-\d{2}-\d{2}T", line), line
        assert re.fullmatch(r"[0-9a-f]{7,40}", fields.get("git", "")), line
        assert re.fullmatch(r"[0-9a-f]{64}", fields.get("protocol_sha256", "")), line
        assert fields.get("split") == "test" and fields.get("reason"), line
