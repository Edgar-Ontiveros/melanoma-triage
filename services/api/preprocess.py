"""Preprocesamiento de la API (F6.3): reproduce la transformación de validación de
entrenamiento con numpy puro, a partir de `preprocess.json`. Ninguna constante aquí.

La cadena de entrenamiento es albumentations: ``SmallestMaxSize(224, cv2.INTER_CUBIC)`` →
``CenterCrop(224)`` → ``Normalize(mean, std, 255)`` → CHW. ``PIL.Image.resize`` no sirve
para reproducirla: PIL aplica antialias al reducir y cv2 no, y la diferencia de logit llega
a ±2. Por eso el redimensionado es una convolución cúbica separable escrita en numpy con la
misma convención que OpenCV (a = −0.75, centros de píxel desplazados medio píxel, sin
antialias, bordes replicados, coeficientes en float32 como ``interpolateCubic``): coincide
con cv2 salvo en el redondeo de ~1 de cada 100,000 píxeles por ±1 (F6.2 mide el efecto
sobre el logit).
"""

from __future__ import annotations

from typing import Any

import numpy as np

_CUBIC_A = -0.75  # coeficiente de la cúbica de Keys que usa OpenCV en INTER_CUBIC
_SUPPORTED_INTERPOLATION = ("bicubic",)
_SUPPORTED_RESIZE = ("center_crop",)


def _cubic_coeffs(frac: np.ndarray, a: float = _CUBIC_A) -> np.ndarray:
    """Los cuatro pesos de la cúbica de Keys para la fracción ``frac`` ∈ [0, 1), en el mismo
    orden de operaciones que ``interpolateCubic`` de OpenCV (el cuarto peso se obtiene por
    diferencia, así los cuatro suman exactamente 1)."""
    x = frac.astype(np.float32)  # OpenCV calcula los coeficientes en float32
    c0 = ((a * (x + 1) - 5 * a) * (x + 1) + 8 * a) * (x + 1) - 4 * a
    c1 = ((a + 2) * x - (a + 3)) * x * x + 1
    c2 = ((a + 2) * (1 - x) - (a + 3)) * (1 - x) * (1 - x) + 1
    c3 = 1 - c0 - c1 - c2
    return np.stack([c0, c1, c2, c3], axis=1).astype(np.float64)


def _resize_axis(img: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    n_in = img.shape[axis]
    scale = n_in / n_out
    src = (np.arange(n_out, dtype=np.float64) + 0.5) * scale - 0.5
    i0 = np.floor(src).astype(np.int64)
    frac = src - i0
    idx = np.clip(np.stack([i0 - 1, i0, i0 + 1, i0 + 2], axis=1), 0, n_in - 1)
    w = _cubic_coeffs(frac)
    gathered = np.take(img, idx, axis=axis)  # ... × n_out × 4 × ...
    shape = [1] * gathered.ndim
    shape[axis], shape[axis + 1] = n_out, 4
    return (gathered * w.reshape(shape)).sum(axis=axis + 1)


def resize_smallest_side(img: np.ndarray, size: int) -> np.ndarray:
    """Lado corto a ``size`` conservando la relación de aspecto; ``round`` como albumentations."""
    h, w = img.shape[:2]
    s = size / min(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    if (nh, nw) == (h, w):
        return img
    out = _resize_axis(img.astype(np.float64), nh, 0)
    out = _resize_axis(out, nw, 1)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    y = (h - size) // 2
    x = (w - size) // 2
    return img[y : y + size, x : x + size]


class Preprocessor:
    """``RGB uint8 [H, W, 3]`` → recorte ``uint8 [S, S, 3]`` y tensor ``float32 [1, 3, S, S]``."""

    def __init__(self, spec: dict[str, Any]) -> None:
        if spec["interpolation"] not in _SUPPORTED_INTERPOLATION:
            raise ValueError(f"interpolación no soportada: {spec['interpolation']!r}")
        if spec["val_resize"] not in _SUPPORTED_RESIZE:
            raise ValueError(f"val_resize no soportado: {spec['val_resize']!r}")
        self.size = int(spec["input_size"])
        self.mean = np.asarray(spec["mean"], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(spec["std"], dtype=np.float32).reshape(1, 1, 3)
        self.pixel_max = float(spec["pixel_max"])

    def crop(self, img: np.ndarray) -> np.ndarray:
        if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
            raise ValueError("se esperaba RGB uint8 [H, W, 3]")
        return center_crop(resize_smallest_side(img, self.size), self.size)

    def tensor(self, crop: np.ndarray) -> np.ndarray:
        x = (crop.astype(np.float32) / self.pixel_max - self.mean) / self.std
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None]).astype(np.float32)

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        crop = self.crop(img)
        return crop, self.tensor(crop)
