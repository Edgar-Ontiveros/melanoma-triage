"""Inferencia en CPU con un checkpoint de F3 sobre un split del manifiesto o sobre un
DataFrame arbitrario de imágenes (F4.0, F4.4, F4.6). Devuelve logit y probabilidad cruda por
imagen junto con la identidad y los metadatos, sin aplicar calibración."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from melanoma.data.dataset import ImageDataset
from melanoma.data.transforms import build_transforms
from melanoma.models import build_model
from melanoma.train.module import LitBinaryClassifier

META_COLUMNS = ["sex", "age_approx", "anatom_site"]


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_checkpoint(
    checkpoint: Path | str, backbone: str, num_classes: int, dropout: float
) -> tuple[LitBinaryClassifier, dict]:
    """Reconstruye el modelo con la fábrica (sin descargar pesos) y carga el checkpoint."""
    model, data_config = build_model(
        backbone=backbone, pretrained=False, num_classes=num_classes, dropout=dropout
    )
    lit = LitBinaryClassifier.load_from_checkpoint(str(checkpoint), model=model, map_location="cpu")
    lit.eval()
    return lit, data_config


@torch.no_grad()
def predict_frame(
    lit: LitBinaryClassifier,
    data_config: dict,
    records: pd.DataFrame,
    images_dir: Path | str,
    image_size: int,
    val_resize: str,
    batch_size: int = 32,
    num_workers: int = 0,
    file_column: str | None = None,
) -> pd.DataFrame:
    """``records`` necesita ``image_id``, ``target`` y ``patient_id``; las imágenes se leen
    como ``images_dir / f"{image_id}.jpg"`` salvo que ``file_column`` dé el nombre de archivo.
    Transformación de validación (sin aumentación). Devuelve ``records`` + ``logit`` +
    ``prob_raw`` en el mismo orden."""
    transform = build_transforms(data_config, image_size, None, train=False, val_resize=val_resize)
    ds = ImageDataset(records, images_dir, transform)
    if file_column is not None:
        files = records[file_column].astype(str).tolist()
        ds.path_of = lambda i, _f=files, _d=Path(images_dir): _d / _f[i]  # type: ignore[method-assign]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    logits = []
    for x, _ in loader:
        logits.append(lit(x).squeeze(1).float().cpu())
    z = torch.cat(logits).numpy().astype(np.float64)
    out = records.reset_index(drop=True).copy()
    out["logit"] = z
    out["prob_raw"] = 1.0 / (1.0 + np.exp(-z))
    return out


def split_records(manifest: pd.DataFrame, image_ids: list[str]) -> pd.DataFrame:
    by_id = manifest.set_index("image_id")
    cols = ["target", "patient_id"] + [c for c in META_COLUMNS if c in by_id.columns]
    recs = by_id.loc[image_ids, cols].reset_index()
    recs["target"] = recs["target"].astype(int)
    return recs
