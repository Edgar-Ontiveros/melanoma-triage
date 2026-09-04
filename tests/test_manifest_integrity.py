"""El manifiesto versionado tiene las filas y positivos publicados (se salta sin datos)."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig

from melanoma.data.manifest import MANIFEST_COLUMNS, official_counts, read_manifest


def test_manifest_integrity(root_dir: Path, prepare_cfg: DictConfig) -> None:
    path = root_dir / prepare_cfg.data.manifest_path
    if not path.exists():
        pytest.skip(f"no existe {path}")
    m = read_manifest(path)
    exp = prepare_cfg.data.expected
    assert list(m.columns[: len(MANIFEST_COLUMNS)]) == MANIFEST_COLUMNS
    got = official_counts(m)
    assert got["images"] == exp.images
    assert got["positives"] == exp.positives
    assert got["patients"] == exp.patients
    assert m["image_id"].is_unique
    assert m["sha256_original"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert set(m["target"].unique()) == {0, 1}
