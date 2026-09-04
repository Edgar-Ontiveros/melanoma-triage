"""Pruebas unitarias del pipeline de datos con datos sintéticos (sin descargas)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from melanoma.data.dedup import (
    UnionFind,
    classify_groups,
    groups_from_pairs,
    pairwise_hamming,
)
from melanoma.data.splits import (
    assign_splits,
    patients_in_two_splits,
    split_summary,
    units_from_cross_groups,
)


def _synthetic_manifest(seed: int = 0, patients: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    k = 0
    for p in range(patients):
        n_img = int(rng.integers(1, 40))
        positive = rng.random() < 0.2
        for _ in range(n_img):
            target = int(positive and rng.random() < 0.3)
            rows.append({"image_id": f"IMG_{k:05d}", "patient_id": f"P_{p:04d}", "target": target})
            k += 1
    return pd.DataFrame(rows)


def test_assign_splits_disjoint_and_covering() -> None:
    m = _synthetic_manifest()
    fractions = {"train": 0.7, "val": 0.15, "test": 0.15}
    split = assign_splits(m, fractions, seed=1)
    assert set(split.index) == set(m["image_id"])
    assert patients_in_two_splits(m, split) == []
    summary = split_summary(m, split, list(fractions))
    total = summary["imagenes"].sum()
    for name, frac in fractions.items():
        got = summary.loc[summary["split"] == name, "imagenes"].iloc[0] / total
        assert abs(got - frac) < 0.05, f"{name}: {got:.3f} vs {frac}"
    pos = summary.set_index("split")["melanomas"]
    assert pos["test"] >= 0.12 * pos.sum()


def test_assign_splits_deterministic() -> None:
    m = _synthetic_manifest()
    fractions = {"train": 0.7, "val": 0.15, "test": 0.15}
    a = assign_splits(m, fractions, seed=7)
    b = assign_splits(m, fractions, seed=7)
    c = assign_splits(m, fractions, seed=8)
    assert a.equals(b)
    assert not a.equals(c)


def test_assign_splits_rejects_bad_fractions() -> None:
    with pytest.raises(ValueError):
        assign_splits(_synthetic_manifest(), {"train": 0.5, "val": 0.1, "test": 0.1}, seed=0)


def test_cross_groups_share_split() -> None:
    m = _synthetic_manifest()
    patient_of = dict(zip(m["image_id"], m["patient_id"], strict=True))
    ids = m["image_id"].tolist()
    # dos grupos cruzados que encadenan tres pacientes distintos
    a = m[m["patient_id"] == "P_0000"]["image_id"].iloc[0]
    b = m[m["patient_id"] == "P_0001"]["image_id"].iloc[0]
    c = m[m["patient_id"] == "P_0002"]["image_id"].iloc[0]
    cross = [[a, b], [b, c]]
    units = units_from_cross_groups(m["patient_id"], cross, patient_of)
    assert units["P_0000"] == units["P_0001"] == units["P_0002"]
    assert units["P_0003"] == "P_0003"
    split = assign_splits(
        m, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=3, unit_of_patient=units
    )
    assert len({split[a], split[b], split[c]}) == 1
    assert patients_in_two_splits(m, split) == []
    assert ids  # el manifiesto sintético no está vacío


def test_pairwise_hamming_and_groups() -> None:
    h = np.array([0b0000, 0b0001, 0b0011, 0b1111 << 60, 0b0000], dtype=np.uint64)
    hist, pairs = pairwise_hamming(h, max_distance=2, block=2)
    assert hist.sum() == 10  # 5 elementos → 10 pares
    assert hist[0] == 1  # (0, 4) son idénticos
    close = {(int(i), int(j)): int(d) for i, j, d in pairs.itertuples(index=False)}
    assert close[(0, 1)] == 1 and close[(1, 2)] == 1 and close[(0, 2)] == 2
    assert (0, 3) not in close
    ids = ["a", "b", "c", "d", "e"]
    groups = groups_from_pairs(pairs, len(ids), ids)
    assert groups == [["a", "b", "c", "e"]]
    intra, cross = classify_groups(groups, {"a": "p1", "b": "p1", "c": "p1", "e": "p2"})
    assert intra == [] and cross == groups


def test_union_find() -> None:
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(3, 4)
    uf.union(1, 4)
    assert uf.groups() == [[0, 1, 3, 4]]
