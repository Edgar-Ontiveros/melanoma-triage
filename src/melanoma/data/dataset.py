"""Dataset de imágenes JPEG redimensionadas con etiqueta binaria (F2.1)."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Una fila del manifiesto → ``(tensor CxHxW, etiqueta)``.

    ``records`` debe tener las columnas ``image_id``, ``target`` y ``patient_id``; las
    imágenes se leen como ``images_dir / f"{image_id}.jpg"``. El orden de ``records`` es el
    orden de iteración cuando el DataLoader no baraja, lo que permite alinear las
    probabilidades predichas con ``patient_id`` para la evaluación.
    """

    def __init__(self, records: pd.DataFrame, images_dir: Path | str, transform: A.Compose):
        self.records = records.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.image_ids: list[str] = self.records["image_id"].astype(str).tolist()
        self.targets: np.ndarray = self.records["target"].to_numpy(dtype=np.int64)
        self.patient_ids: list[str] = self.records["patient_id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.records)

    def path_of(self, index: int) -> Path:
        return self.images_dir / f"{self.image_ids[index]}.jpg"

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.path_of(index)) as img:
            array = np.asarray(img.convert("RGB"))
        x = self.transform(image=array)["image"]
        return x, torch.tensor(self.targets[index], dtype=torch.long)

    def class_counts(self) -> tuple[int, int]:
        """``(negativos, positivos)``."""
        pos = int(self.targets.sum())
        return len(self.targets) - pos, pos
