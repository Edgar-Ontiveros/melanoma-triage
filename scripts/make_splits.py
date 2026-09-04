"""F1.4 — Splits agrupados por paciente: ``data/splits/{train,val,test}.txt``.

Lee el manifiesto y los grupos cruzados de ``reports/dedup_cross_groups.json`` (si existe),
fusiona los pacientes de cada grupo cruzado en una unidad, reparte 70/15/15 estratificando
a nivel paciente por "tiene al menos un melanoma", y escribe un ``image_id`` por línea,
ordenado. Además: ``data/splits/SHA256SUMS`` y ``reports/splits_report.md``.

Se detiene si el conjunto de prueba queda por debajo de ``data.split.min_test_positives``.

Uso: ``uv run python scripts/make_splits.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from melanoma.data.hashing import sha256_file
from melanoma.data.manifest import read_manifest
from melanoma.data.splits import (
    assign_splits,
    patients_in_two_splits,
    split_summary,
    units_from_cross_groups,
)


@hydra.main(config_path="../configs", config_name="prepare", version_base="1.3")
def main(cfg: DictConfig) -> None:
    d = cfg.data
    names = list(d.split.names)
    fractions = {n: float(d.split.fractions[n]) for n in names}
    m = read_manifest(d.manifest_path)
    patient_of = dict(zip(m["image_id"], m["patient_id"], strict=True))

    cross_path = Path(d.reports_dir) / "dedup_cross_groups.json"
    cross_groups: list[list[str]] = []
    if cross_path.exists():
        cross_groups = json.loads(cross_path.read_text(encoding="utf-8"))["cross_groups"]
    else:
        print(f"AVISO: no existe {cross_path}; se asume que no hay duplicados cruzados")
    units = units_from_cross_groups(m["patient_id"], cross_groups, patient_of)
    merged_patients = sum(1 for p, u in units.items() if p != u)

    split = assign_splits(m, fractions, int(d.split.seed), units)
    leaks = patients_in_two_splits(m, split)
    if leaks:
        sys.exit(f"ERROR: pacientes en dos splits: {leaks[:10]}")

    summary = split_summary(m, split, names)
    test_name = names[-1]
    test_pos = int(summary.loc[summary["split"] == test_name, "melanomas"].iloc[0])

    splits_dir = Path(d.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    sums = []
    for name in names:
        ids = sorted(split[split == name].index.tolist())
        path = splits_dir / f"{name}.txt"
        path.write_text("\n".join(ids) + "\n", encoding="utf-8")
        sums.append(f"{sha256_file(path)}  {path.name}")
    (splits_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")

    lines = [
        "# Splits ISIC 2020 — conteos alcanzados",
        "",
        f"Generado por `scripts/make_splits.py`. Semilla `{d.split.seed}`, fracciones "
        f"{fractions} (`configs/data/isic2020.yaml`).",
        "",
        "Unidad de agrupación: `patient_id`. Estratificación a nivel paciente por "
        "«tiene al menos un melanoma». Reparto voraz por déficit: el estrato positivo se "
        "balancea por número de melanomas y el negativo por número de imágenes.",
        "",
        f"- Pacientes: {m['patient_id'].nunique():,}; unidades de split tras fusionar "
        f"grupos cruzados: {len(set(units.values())):,} "
        f"({len(cross_groups)} grupos cruzados, {merged_patients} pacientes fusionados)",
        "- Imágenes excluidas del conjunto de prueba por duplicados cruzados: 0 "
        "(la fusión de pacientes en unidades hace innecesaria la exclusión)",
        f"- Pacientes en más de un split: {len(leaks)}",
        "",
        summary.to_markdown(index=False),
        "",
        f"Restricción: el conjunto de prueba debe tener ≥ {d.split.min_test_positives} "
        f"melanomas. Obtenidos: **{test_pos}**.",
        "",
        "## SHA256 de los archivos de split",
        "",
        "```",
        *sums,
        "```",
        "",
    ]
    report = Path(d.reports_dir) / "splits_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\n".join(sums))
    print(OmegaConf.to_yaml(d.split))
    if test_pos < d.split.min_test_positives:
        sys.exit(
            f"ERROR: {test_pos} melanomas en prueba < {d.split.min_test_positives}. "
            "Detener y decidir la proporción (spec F1.4)."
        )
    print(f"F1.4 OK. Reporte: {report}")


if __name__ == "__main__":
    main()
