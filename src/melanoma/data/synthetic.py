"""Dataset sintético en disco para la prueba de humo y las pruebas unitarias.

Escribe JPEG de ruido con la misma estructura que ``data/processed/isic2020_512`` (un archivo
``{image_id}.jpg``), un manifiesto con las columnas que usa el DataModule y la línea base de
metadatos, y los splits de entrenamiento y validación con su ``SHA256SUMS``. No descarga nada.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SPLIT_NAMES = ("train", "val")
SITES = ("torso", "lower extremity", "upper extremity", "head/neck")


def write_synthetic_dataset(
    root: Path | str,
    n_images: int,
    positive_rate: float,
    seed: int,
    size: tuple[int, int] = (96, 72),
    images_per_patient: int = 4,
    val_fraction: float = 0.3,
) -> dict[str, Path]:
    """Crea ``images/``, ``manifest.csv`` y ``splits/{train,val}.txt`` bajo ``root``.

    Los positivos tienen un tinte distinto al de los negativos para que un modelo pueda
    aprender algo en dos épocas; no pretende parecerse a dermatoscopia.
    """
    root = Path(root)
    images_dir = root / "images"
    splits_dir = root / "splits"
    images_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_images):
        image_id = f"SYN_{k:06d}"
        patient = f"SP_{k // images_per_patient:05d}"
        target = int(rng.random() < positive_rate)
        base = np.array([200, 120, 120] if target else [120, 120, 200], dtype=np.float64)
        noise = rng.normal(0, 30, size=(size[1], size[0], 3))
        pixels = np.clip(base + noise, 0, 255).astype(np.uint8)
        Image.fromarray(pixels).save(images_dir / f"{image_id}.jpg", quality=90)
        rows.append(
            {
                "image_id": image_id,
                "patient_id": patient,
                "sex": rng.choice(["male", "female", None], p=[0.49, 0.49, 0.02]),
                "age_approx": float(rng.choice([30, 45, 60, 75])) if rng.random() > 0.02 else None,
                "anatom_site": rng.choice(list(SITES) + [None], p=[0.3, 0.3, 0.2, 0.18, 0.02]),
                "target": target,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest_path = root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    patients = rng.permutation(manifest["patient_id"].unique().astype(str))
    n_val = max(1, int(round(val_fraction * len(patients))))
    val_patients = set(patients[:n_val])
    ids = {
        "train": sorted(manifest.loc[~manifest["patient_id"].isin(val_patients), "image_id"]),
        "val": sorted(manifest.loc[manifest["patient_id"].isin(val_patients), "image_id"]),
    }
    sums = []
    for name in SPLIT_NAMES:
        path = splits_dir / f"{name}.txt"
        path.write_text("\n".join(ids[name]) + "\n", encoding="utf-8")
        sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}.txt")
    (splits_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return {"images_dir": images_dir, "manifest_path": manifest_path, "splits_dir": splits_dir}
