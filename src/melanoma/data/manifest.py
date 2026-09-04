"""Manifiesto de ISIC 2020: una fila por imagen con huella, dimensiones y metadatos.

Columnas (spec F1.2): ``image_id, sha256_original, relpath_original, width, height, bytes,
patient_id, lesion_id, sex, age_approx, anatom_site, target, diagnosis``. Las columnas
``sha256_resized`` y ``relpath_resized`` se agregan en F1.5.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from melanoma.data.hashing import fingerprint_many

MANIFEST_COLUMNS = [
    "image_id",
    "sha256_original",
    "relpath_original",
    "width",
    "height",
    "bytes",
    "patient_id",
    "lesion_id",
    "sex",
    "age_approx",
    "anatom_site",
    "target",
    "diagnosis",
]

# Nombres de columna del CSV oficial (GroundTruth_v2) → nombres del manifiesto.
_GT_RENAME = {
    "image_name": "image_id",
    "anatom_site_general_challenge": "anatom_site",
}


def load_ground_truth(csv_path: Path | str) -> pd.DataFrame:
    """Lee el ground truth oficial y normaliza nombres de columna."""
    gt = pd.read_csv(csv_path)
    gt = gt.rename(columns=_GT_RENAME)
    gt["target"] = gt["target"].astype(int)
    return gt


def official_counts(df: pd.DataFrame) -> dict[str, float]:
    """Conteos que se contrastan con la publicación oficial."""
    n = len(df)
    pos = int(df["target"].sum())
    return {
        "images": n,
        "positives": pos,
        "patients": int(df["patient_id"].nunique()),
        "prevalence_pct": round(100.0 * pos / n, 2) if n else 0.0,
    }


def verify_official_counts(df: pd.DataFrame, expected: dict[str, float]) -> pd.DataFrame:
    """Tabla ``cantidad, esperado, obtenido, ok``. La fase se detiene si alguna fila falla."""
    got = official_counts(df)
    rows = []
    for key, exp in expected.items():
        val = got[key]
        ok = abs(val - exp) < 0.005 if key == "prevalence_pct" else val == exp
        rows.append({"cantidad": key, "esperado": exp, "obtenido": val, "ok": bool(ok)})
    return pd.DataFrame(rows)


def build_manifest(
    gt: pd.DataFrame, images_dir: Path | str, root: Path | str, workers: int
) -> pd.DataFrame:
    """Une el ground truth con la huella y dimensiones de cada archivo JPEG.

    Falla si falta algún archivo del CSV o si hay archivos en disco que no están en el CSV:
    ambas cosas son discrepancias con la publicación oficial y hay que reportarlas.
    """
    images_dir = Path(images_dir)
    root = Path(root)
    on_disk = {p.stem: p for p in images_dir.glob("*.jpg")}
    in_csv = set(gt["image_id"])
    missing = sorted(in_csv - set(on_disk))
    extra = sorted(set(on_disk) - in_csv)
    if missing or extra:
        raise RuntimeError(
            f"discrepancia CSV/disco: faltan {len(missing)} archivos {missing[:5]}, "
            f"sobran {len(extra)} {extra[:5]}"
        )
    paths = [on_disk[i] for i in gt["image_id"]]
    fps = fingerprint_many(paths, workers=workers)
    out = gt.copy()
    out["sha256_original"] = [fp["sha256"] for fp in fps]
    out["relpath_original"] = [str(p.relative_to(root)) for p in paths]
    out["width"] = [fp["width"] for fp in fps]
    out["height"] = [fp["height"] for fp in fps]
    out["bytes"] = [fp["bytes"] for fp in fps]
    return out[MANIFEST_COLUMNS].sort_values("image_id").reset_index(drop=True)


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Faltantes por columna: cantidad y porcentaje sobre el total de filas."""
    n = len(df)
    na = df.isna().sum()
    return pd.DataFrame(
        {"columna": na.index, "faltantes": na.values, "pct": (100.0 * na.values / n).round(2)}
    )


def read_manifest(path: Path | str) -> pd.DataFrame:
    """Lee el manifiesto versionado, con ``image_id`` y ``patient_id`` como texto."""
    return pd.read_csv(path, dtype={"image_id": str, "patient_id": str, "lesion_id": str})
