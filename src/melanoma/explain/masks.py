"""Máscara automática y cruda de la lesión (F5.3). ISIC 2020 no trae segmentación.

Receta fija de la spec: escala de grises → Otsu (la lesión es más oscura que la piel) →
apertura y cierre morfológicos → componente conexo más grande que no toque más del 30 % del
borde (y que alcance el cuadrado central, para no confundir una esquina oscura con la lesión).
Filtro de calidad: área entre 3 % y 70 % de la imagen. Es una heurística, no un
segmentador; el reporte muestra ejemplos buenos y malos y declara cuántas imágenes pasan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

MIN_AREA_FRAC = 0.03
MAX_AREA_FRAC = 0.70
MAX_BORDER_FRAC = 0.30
KERNEL_FRAC = 0.03  # diámetro del elemento estructurante como fracción del lado


@dataclass(frozen=True)
class MaskInfo:
    valid: bool
    reason: str
    area_frac: float
    border_frac: float
    n_components: int
    otsu_threshold: float

    def as_dict(self) -> dict:
        return asdict(self)


def _border_fraction(component: np.ndarray) -> float:
    """Fracción de los píxeles del perímetro de la imagen que pertenecen a la componente."""
    border = np.concatenate(
        [component[0, :], component[-1, :], component[1:-1, 0], component[1:-1, -1]]
    )
    return float(border.mean()) if border.size else 0.0


def lesion_mask(image_rgb: np.ndarray) -> tuple[np.ndarray, MaskInfo]:
    """``(mask booleana [H, W], MaskInfo)`` para una imagen RGB ``uint8``.

    Si no hay componente válida la máscara devuelta es toda ``False`` y ``info.valid`` es
    ``False`` con la razón (``no_component``, ``too_small``, ``too_large``).
    """
    img = np.asarray(image_rgb)
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError(f"se esperaba RGB uint8 [H, W, 3]; llegó {img.shape} {img.dtype}")
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thr, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = max(3, int(round(KERNEL_FRAC * min(h, w))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    empty = np.zeros((h, w), dtype=bool)
    # Etiqueta 0 es el fondo. Candidatas: componentes que tocan ≤ 30 % del perímetro Y que
    # alcanzan el cuadrado central de la imagen (del 25 % al 75 % de cada lado). Lo segundo
    # descarta las esquinas oscuras de viñeta, que pueden ser mayores que una lesión pequeña y
    # tocar menos del 30 % del perímetro; una lesión dermatoscópica real está centrada.
    center = np.zeros((h, w), dtype=bool)
    center[h // 4 : (3 * h) // 4, w // 4 : (3 * w) // 4] = True
    candidates = []
    for label in range(1, n_labels):
        comp = labels == label
        border = _border_fraction(comp)
        if border <= MAX_BORDER_FRAC and comp[center].any():
            candidates.append((float(comp.mean()), border, label))
    pool = candidates
    for area, border, label in sorted(pool, key=lambda c: -c[0]):
        comp = labels == label
        if area < MIN_AREA_FRAC:
            reason, valid = "too_small", False
        elif area > MAX_AREA_FRAC:
            reason, valid = "too_large", False
        else:
            reason, valid = "ok", True
        info = MaskInfo(valid, reason, area, border, n_labels - 1, float(thr))
        return (comp if valid else empty), info
    return empty, MaskInfo(False, "no_component", 0.0, 0.0, n_labels - 1, float(thr))


def mask_centroid_radius(mask: np.ndarray) -> tuple[float, float, float]:
    """``(cy, cx, radio equivalente)`` de una máscara; centro de la imagen y un cuarto del
    lado si la máscara está vacía (para colocar artefactos «cerca de la lesión»)."""
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    if not m.any():
        return h / 2.0, w / 2.0, 0.25 * min(h, w)
    ys, xs = np.nonzero(m)
    return float(ys.mean()), float(xs.mean()), float(np.sqrt(m.sum() / np.pi))
