"""Preprocesamiento de la API (F6.3): reproduce las DOS etapas por las que pasaron las
imágenes de entrenamiento, con los parámetros de `preprocess.json`. Ninguna constante aquí.

Etapa 1 — la de F1 (`melanoma.data.resize.resize_one`): decodificación reducida de PIL
(``draft``), lado largo a ``stage1.long_side`` con ``stage1.filter`` (Lanczos) sin agrandar
nunca, y reencodificación JPEG a ``stage1.jpeg_quality`` (el modelo vio los píxeles ya
recomprimidos; se replica en memoria). Si la imagen ya tiene lado largo ≤ ``long_side`` la
etapa no hace nada: es el caso de las imágenes de ``data/processed`` y de las pruebas de
paridad a 512 px.

Etapa 2 — la de entrenamiento (albumentations): ``SmallestMaxSize(input_size, INTER_CUBIC)``
→ ``CenterCrop`` → ``Normalize(mean, std, 255)`` → CHW. El redimensionado usa
``cv2.resize`` (opencv-python-headless en el grupo ``api``): es la misma función que usó
albumentations, así que la paridad es exacta.

Un solo cúbico de 6,000 a 224 px sin la etapa 1 produce aliasing que el modelo nunca vio;
por eso las dos etapas.
"""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
from PIL import Image

_INTERPOLATION = {"bicubic": cv2.INTER_CUBIC, "bilinear": cv2.INTER_LINEAR}
_PIL_FILTERS = {
    "lanczos": Image.Resampling.LANCZOS,
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
}
_SUPPORTED_RESIZE = ("center_crop",)


def stage1_resize(
    img: Image.Image, long_side: int, pil_filter: str, jpeg_quality: int
) -> Image.Image:
    """Réplica de ``resize_one`` de F1 sobre una imagen PIL abierta (sin ``load``): ``draft``
    → RGB → lado largo a ``long_side`` si es mayor → JPEG en memoria a ``jpeg_quality``.
    Si el lado largo ya es ≤ ``long_side`` devuelve la imagen en RGB sin tocarla."""
    if max(img.size) <= long_side:
        return img.convert("RGB")
    img.draft("RGB", (long_side, long_side))  # decodificación reducida, nunca < long_side
    img = img.convert("RGB")
    w, h = img.size
    scale = long_side / max(w, h)
    if scale < 1.0:
        img = img.resize((round(w * scale), round(h * scale)), _PIL_FILTERS[pil_filter])
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(jpeg_quality))
    buf.seek(0)
    with Image.open(buf) as rt:
        return rt.convert("RGB")


def resize_smallest_side(img: np.ndarray, size: int, interpolation: int) -> np.ndarray:
    """Lado corto a ``size`` conservando la relación de aspecto; ``round`` y ``cv2.resize``
    como ``SmallestMaxSize`` de albumentations."""
    h, w = img.shape[:2]
    s = size / min(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    if (nh, nw) == (h, w):
        return img
    return cv2.resize(img, (nw, nh), interpolation=interpolation)


def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    y = (h - size) // 2
    x = (w - size) // 2
    return img[y : y + size, x : x + size]


class Preprocessor:
    """PIL → (etapa 1) → RGB uint8 → (etapa 2) recorte ``uint8 [S, S, 3]`` y tensor
    ``float32 [1, 3, S, S]``."""

    def __init__(self, spec: dict[str, Any]) -> None:
        if spec["interpolation"] not in _INTERPOLATION:
            raise ValueError(f"interpolación no soportada: {spec['interpolation']!r}")
        if spec["val_resize"] not in _SUPPORTED_RESIZE:
            raise ValueError(f"val_resize no soportado: {spec['val_resize']!r}")
        s1 = spec["stage1"]
        if s1["filter"] not in _PIL_FILTERS:
            raise ValueError(f"filtro de la etapa 1 no soportado: {s1['filter']!r}")
        self.size = int(spec["input_size"])
        self.interpolation = _INTERPOLATION[spec["interpolation"]]
        self.mean = np.asarray(spec["mean"], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(spec["std"], dtype=np.float32).reshape(1, 1, 3)
        self.pixel_max = float(spec["pixel_max"])
        self.long_side = int(s1["long_side"])
        self.pil_filter = str(s1["filter"])
        self.jpeg_quality = int(s1["jpeg_quality"])

    def stage1(self, img: Image.Image) -> np.ndarray:
        return np.asarray(stage1_resize(img, self.long_side, self.pil_filter, self.jpeg_quality))

    def crop(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("se esperaba RGB uint8 [H, W, 3]")
        return center_crop(resize_smallest_side(rgb, self.size, self.interpolation), self.size)

    def tensor(self, crop: np.ndarray) -> np.ndarray:
        x = (crop.astype(np.float32) / self.pixel_max - self.mean) / self.std
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None]).astype(np.float32)

    def from_array(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Solo la etapa 2 (la entrada ya es ≤ ``long_side`` o ya pasó por la etapa 1)."""
        crop = self.crop(rgb)
        return crop, self.tensor(crop)

    def from_pil(self, img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Las dos etapas: ``img`` recién abierta (sin decodificar del todo, para ``draft``)."""
        return self.from_array(self.stage1(img))

    __call__ = from_array
