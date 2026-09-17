"""DataModule de Lightning para ISIC 2020 (F2.1).

Contrato: manifiesto + directorio de splits + directorio de imágenes + ``data_config`` de la
fábrica → DataLoaders de entrenamiento y validación. El de prueba está deshabilitado: pedirlo
lanza ``LockedTestSplitError``. El único acceso al split de prueba en todo el proyecto es
``scripts/evaluate_test.py`` (F4).

Solo entrenamiento se aumenta y, si se pide, se rebalancea con ``WeightedRandomSampler``.
Validación se sirve en orden, sin aumentación ni remuestreo.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from melanoma.data.dataset import ImageDataset
from melanoma.data.manifest import read_manifest
from melanoma.data.transforms import build_transforms

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
SAMPLERS = ("none", "weighted")


class LockedTestSplitError(RuntimeError):
    """El split de prueba no se carga desde el DataModule (bloqueado hasta F4)."""


def read_split_ids(splits_dir: Path | str, name: str) -> list[str]:
    path = Path(splits_dir) / f"{name}.txt"
    ids = path.read_text(encoding="utf-8").split()
    if not ids:
        raise ValueError(f"split vacío: {path}")
    return ids


def split_file_sha256(splits_dir: Path | str, name: str) -> str:
    return hashlib.sha256((Path(splits_dir) / f"{name}.txt").read_bytes()).hexdigest()


def verify_split_hashes(splits_dir: Path | str, names: tuple[str, ...] = (TRAIN_SPLIT, VAL_SPLIT)):
    """Compara los archivos de split con ``SHA256SUMS``. Falla si alguno no coincide.

    Returns:
        ``{nombre: sha256}`` de los archivos verificados.
    """
    splits_dir = Path(splits_dir)
    listed: dict[str, str] = {}
    for line in (splits_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            listed[parts[1]] = parts[0]
    out: dict[str, str] = {}
    for name in names:
        actual = split_file_sha256(splits_dir, name)
        expected = listed.get(f"{name}.txt")
        if expected != actual:
            raise RuntimeError(
                f"{name}.txt no coincide con SHA256SUMS (esperado {expected}, obtenido {actual})"
            )
        out[name] = actual
    return out


class ISICDataModule(L.LightningDataModule):
    def __init__(
        self,
        manifest_path: Path | str,
        splits_dir: Path | str,
        images_dir: Path | str,
        data_config: Mapping[str, Any],
        batch_size: int,
        num_workers: int,
        image_size: int | None = None,
        augment: Mapping[str, Any] | None = None,
        sampler: str = "none",
        preprocess_scale: str = "none",
        val_resize: str = "center_crop",
    ) -> None:
        super().__init__()
        if sampler not in SAMPLERS:
            raise ValueError(f"sampler desconocido: {sampler!r}; opciones: {SAMPLERS}")
        self.manifest_path = Path(manifest_path)
        self.splits_dir = Path(splits_dir)
        self.images_dir = Path(images_dir)
        self.data_config = dict(data_config)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.image_size = image_size
        self.augment = dict(augment) if augment is not None else None
        self.sampler = sampler
        self.preprocess_scale = preprocess_scale
        self.val_resize = val_resize
        self.train_ds: ImageDataset | None = None
        self.val_ds: ImageDataset | None = None

    # ---- carga -------------------------------------------------------------------------
    def _records(self, manifest: pd.DataFrame, name: str) -> pd.DataFrame:
        ids = read_split_ids(self.splits_dir, name)
        by_id = manifest.set_index("image_id")
        missing = [i for i in ids if i not in by_id.index]
        if missing:
            raise KeyError(
                f"{len(missing)} image_id de {name} no están en el manifiesto: {missing[:5]}"
            )
        recs = by_id.loc[ids, ["target", "patient_id"]].reset_index()
        recs["target"] = recs["target"].astype(int)
        return recs

    def setup(self, stage: str | None = None) -> None:
        if stage == "test":
            raise LockedTestSplitError(
                "el DataModule no carga el split de prueba; usar scripts/evaluate_test.py (F4)"
            )
        if self.train_ds is not None and self.val_ds is not None:
            return
        manifest = read_manifest(self.manifest_path)
        train_tf = build_transforms(
            self.data_config,
            self.image_size,
            self.augment,
            train=True,
            preprocess_scale=self.preprocess_scale,
        )
        val_tf = build_transforms(
            self.data_config,
            self.image_size,
            None,
            train=False,
            preprocess_scale=self.preprocess_scale,
            val_resize=self.val_resize,
        )
        self.train_ds = ImageDataset(
            self._records(manifest, TRAIN_SPLIT), self.images_dir, train_tf
        )
        self.val_ds = ImageDataset(self._records(manifest, VAL_SPLIT), self.images_dir, val_tf)

    # ---- desbalance ----------------------------------------------------------------------
    def class_counts(self) -> tuple[int, int]:
        """``(negativos, positivos)`` del split de entrenamiento."""
        assert self.train_ds is not None, "llamar setup() antes"
        return self.train_ds.class_counts()

    def pos_weight(self) -> float:
        """Razón negativos/positivos de entrenamiento, para ``BCEWithLogitsLoss``."""
        neg, pos = self.class_counts()
        if pos == 0:
            raise ValueError("el split de entrenamiento no tiene positivos")
        return neg / pos

    def _train_sampler(self) -> WeightedRandomSampler | None:
        if self.sampler != "weighted":
            return None
        assert self.train_ds is not None
        neg, pos = self.class_counts()
        counts = torch.tensor([neg, pos], dtype=torch.float64)
        weights = 1.0 / counts[torch.from_numpy(self.train_ds.targets.copy())]
        return WeightedRandomSampler(weights, num_samples=len(self.train_ds), replacement=True)

    # ---- loaders -------------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None, "llamar setup() antes"
        sampler = self._train_sampler()
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None, "llamar setup() antes"
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        raise LockedTestSplitError(
            "el split de prueba está bloqueado hasta F4 (scripts/evaluate_test.py); "
            "en F2 y F3 solo existen train y val"
        )

    def predict_dataloader(self) -> DataLoader:
        """Validación en orden, para alinear predicciones con ``patient_id``."""
        return self.val_dataloader()

    def val_frame(self) -> pd.DataFrame:
        """``image_id, target, patient_id`` de validación, en el orden del DataLoader."""
        assert self.val_ds is not None, "llamar setup() antes"
        return self.val_ds.records.copy()

    def split_hashes(self) -> dict[str, str]:
        return {n: split_file_sha256(self.splits_dir, n) for n in (TRAIN_SPLIT, VAL_SPLIT)}


def datamodule_from_config(
    data_cfg: Mapping[str, Any], data_config: Mapping[str, Any], root: Path | str | None = None
) -> ISICDataModule:
    """Construye el DataModule desde el grupo ``data`` de Hydra y el data_config de la fábrica."""
    from melanoma.data.paths import resolve_paths

    paths = resolve_paths(data_cfg, root=root)
    return ISICDataModule(
        manifest_path=paths.manifest_path,
        splits_dir=paths.splits_dir,
        images_dir=paths.images_dir,
        data_config=data_config,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        image_size=data_cfg.get("image_size"),
        augment=data_cfg["augment"],
        sampler=str(data_cfg.get("sampler", "none")),
        preprocess_scale=str(data_cfg.get("preprocess_scale", "none")),
        val_resize=str(data_cfg.get("val_resize", "center_crop")),
    )
