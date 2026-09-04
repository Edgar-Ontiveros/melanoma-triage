"""Splits agrupados por paciente y estratificados a nivel paciente.

Unidad de agrupación: ``patient_id`` (o una *unidad* que fusiona varios pacientes cuando un
grupo de duplicados cruzados los une; así todos los miembros del grupo caen en el mismo
split sin excluir nada del conjunto de prueba). Ninguna imagen de una unidad queda en dos
splits. La estratificación es binaria: la unidad tiene al menos un melanoma o no.

Asignación: dentro de cada estrato las unidades se barajan con la semilla y se reparten
con un criterio voraz de déficit —cada unidad va al split que está más lejos de su cuota—.
El estrato positivo se balancea por número de melanomas (es lo que fija el intervalo de
confianza en F4); el negativo, por número de imágenes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from melanoma.data.dedup import UnionFind


def units_from_cross_groups(
    patient_ids: Sequence[str], cross_groups: list[list[str]], patient_of: Mapping[str, str]
) -> dict[str, str]:
    """``patient_id → unit_id``. Pacientes unidos por un grupo cruzado comparten unidad."""
    patients = sorted(set(patient_ids))
    index = {p: k for k, p in enumerate(patients)}
    uf = UnionFind(len(patients))
    for group in cross_groups:
        members = sorted({patient_of[i] for i in group})
        for other in members[1:]:
            uf.union(index[members[0]], index[other])
    return {p: patients[uf.find(index[p])] for p in patients}


def _greedy_assign(
    weights: pd.Series, fractions: Mapping[str, float], rng: np.random.Generator
) -> dict[str, str]:
    """Reparte las unidades (índice de ``weights``) entre splits siguiendo las cuotas."""
    names = list(fractions)
    total = float(weights.sum())
    target = {s: fractions[s] * total for s in names}
    current = dict.fromkeys(names, 0.0)
    order = weights.index.to_numpy()[rng.permutation(len(weights))]
    out: dict[str, str] = {}
    for unit in order:
        w = float(weights[unit])
        # split con mayor déficit relativo después de recibir la unidad
        best = min(names, key=lambda s: (current[s] + w) / target[s])
        out[unit] = best
        current[best] += w
    return out


def assign_splits(
    manifest: pd.DataFrame,
    fractions: Mapping[str, float],
    seed: int,
    unit_of_patient: Mapping[str, str] | None = None,
) -> pd.Series:
    """Serie ``image_id → split`` (índice ``image_id``)."""
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"las fracciones deben sumar 1: {dict(fractions)}")
    df = manifest[["image_id", "patient_id", "target"]].copy()
    if unit_of_patient is None:
        unit_of_patient = {p: p for p in df["patient_id"].unique()}
    df["unit"] = df["patient_id"].map(unit_of_patient)
    per_unit = df.groupby("unit").agg(images=("image_id", "size"), melanomas=("target", "sum"))
    positive = per_unit[per_unit["melanomas"] > 0]
    negative = per_unit[per_unit["melanomas"] == 0]
    rng = np.random.default_rng(seed)
    assignment = _greedy_assign(positive["melanomas"].astype(float), fractions, rng)
    assignment.update(_greedy_assign(negative["images"].astype(float), fractions, rng))
    split = df["unit"].map(assignment)
    split.index = df["image_id"].to_numpy()
    split.name = "split"
    return split


def split_summary(manifest: pd.DataFrame, split: pd.Series, names: Sequence[str]) -> pd.DataFrame:
    """Conteos alcanzados por split: pacientes, imágenes, melanomas y prevalencia."""
    df = manifest.set_index("image_id").join(split)
    rows = []
    for name in names:
        part = df[df["split"] == name]
        n = len(part)
        pos = int(part["target"].sum())
        rows.append(
            {
                "split": name,
                "pacientes": int(part["patient_id"].nunique()),
                "imagenes": n,
                "melanomas": pos,
                "prevalencia_pct": round(100.0 * pos / n, 2) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def patients_in_two_splits(manifest: pd.DataFrame, split: pd.Series) -> list[str]:
    """Pacientes que aparecen en más de un split. Debe ser una lista vacía."""
    df = manifest.set_index("image_id").join(split)
    n_splits = df.groupby("patient_id")["split"].nunique()
    return sorted(n_splits[n_splits > 1].index.tolist())
