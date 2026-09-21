"""Artefactos sintéticos para el experimento pareado de F5.4.

Cada perturbación recibe una imagen RGB ``uint8`` (el recorte que ve el modelo), un generador
aleatorio y opcionalmente la máscara de la lesión, y devuelve ``(imagen, region)``: la imagen
perturbada con la misma forma y tipo, y una máscara booleana de dónde se puso el artefacto
(para medir si el CAM se movió hacia él). El ruido gaussiano es el control sin estructura:
su región es toda la imagen.

Los parámetros son fijos y están a la vista: no se calibraron contra el modelo.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from melanoma.explain.masks import mask_centroid_radius

Perturbation = Callable[[np.ndarray, np.random.Generator, np.ndarray | None], tuple]

# Colores (RGB). La tinta quirúrgica típica es azul-violeta; el vello, marrón muy oscuro.
INK_RGB = (72, 40, 150)
HAIR_RGB = (28, 18, 12)
RULER_BG_RGB = (235, 232, 225)
RULER_TICK_RGB = (25, 25, 25)

RULER_FRAC = 0.09  # ancho de la franja como fracción del lado
VIGNETTE_STRENGTH = 0.75  # oscurecimiento en la esquina más lejana (0 = nada)
VIGNETTE_INNER = 0.45  # radio normalizado hasta el que la imagen no se toca
VIGNETTE_REGION_FACTOR = 0.85  # píxeles atenuados por debajo de este factor cuentan como región
NOISE_SIGMA = 12.0  # desviación del ruido gaussiano en escala 0–255


def _check(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError(f"se esperaba RGB uint8 [H, W, 3]; llegó {img.shape} {img.dtype}")
    return img


def _blend(image: np.ndarray, layer: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """``image·(1−α) + layer·α`` con ``α`` en ``[0, 1]`` por píxel."""
    a = alpha[..., None].astype(np.float32)
    out = image.astype(np.float32) * (1 - a) + layer.astype(np.float32) * a
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def ruler(
    image: np.ndarray, rng: np.random.Generator, lesion: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Franja clara con marcas de escala en un borde elegido al azar, estilo regla de
    dermatoscopio: marcas cada 1/20 del lado, más largas cada cinco."""
    img = _check(image)
    h, w = img.shape[:2]
    side = int(rng.integers(4))  # 0 abajo, 1 arriba, 2 derecha, 3 izquierda
    band = max(6, int(round(RULER_FRAC * min(h, w))))
    layer = np.zeros_like(img)
    layer[:] = RULER_BG_RGB
    region = np.zeros((h, w), dtype=bool)
    if side in (0, 1):
        rows = slice(h - band, h) if side == 0 else slice(0, band)
        region[rows, :] = True
        step = max(4, w // 20)
        for i, x in enumerate(range(0, w, step)):
            length = band // 2 if i % 5 == 0 else band // 4
            y0, y1 = (h - length, h) if side == 0 else (0, length)
            cv2.line(layer, (x, y0), (x, y1), RULER_TICK_RGB, 1)
    else:
        cols = slice(w - band, w) if side == 2 else slice(0, band)
        region[:, cols] = True
        step = max(4, h // 20)
        for i, y in enumerate(range(0, h, step)):
            length = band // 2 if i % 5 == 0 else band // 4
            x0, x1 = (w - length, w) if side == 2 else (0, length)
            cv2.line(layer, (x0, y), (x1, y), RULER_TICK_RGB, 1)
    return _blend(img, layer, region.astype(np.float32)), region


def ink(
    image: np.ndarray, rng: np.random.Generator, lesion: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Trazo grueso azul-violeta cerca de la lesión sin cubrirla: se coloca a ``radio + margen``
    del centroide de la máscara, en el ángulo (de ocho candidatos) que menos la invade; los
    píxeles de la lesión se excluyen del trazo en cualquier caso."""
    img = _check(image)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=bool) if lesion is None else np.asarray(lesion, dtype=bool)
    cy, cx, r = mask_centroid_radius(mask)
    thickness = max(3, int(round(0.03 * min(h, w))))
    length = int(round(0.22 * min(h, w)))
    dist = r + 0.12 * min(h, w)
    dilated = cv2.dilate(mask.astype(np.uint8), np.ones((thickness, thickness), np.uint8)) > 0
    best, best_overlap = None, None
    for angle in rng.permutation(8) * (np.pi / 4):
        mx = cx + dist * np.cos(angle)
        my = cy + dist * np.sin(angle)
        # Trazo tangencial, ligeramente curvo (tres puntos).
        tx, ty = -np.sin(angle), np.cos(angle)
        pts = np.array(
            [
                [mx - tx * length / 2, my - ty * length / 2],
                [mx + np.cos(angle) * 0.06 * min(h, w), my + np.sin(angle) * 0.06 * min(h, w)],
                [mx + tx * length / 2, my + ty * length / 2],
            ]
        )
        if (pts[:, 0].min() < 0 or pts[:, 0].max() >= w) or (
            pts[:, 1].min() < 0 or pts[:, 1].max() >= h
        ):
            continue
        stroke = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(stroke, [np.rint(pts).astype(np.int32)], False, 255, thickness, cv2.LINE_AA)
        overlap = int((stroke > 0)[dilated].sum())
        if best is None or overlap < best_overlap:
            best, best_overlap = stroke, overlap
        if overlap == 0:
            break
    if best is None:  # lesión enorme o pegada al borde: trazo en la esquina menos cubierta
        best = np.zeros((h, w), dtype=np.uint8)
        cv2.line(best, (thickness, thickness), (length, thickness), 255, thickness, cv2.LINE_AA)
    alpha = best.astype(np.float32) / 255.0 * 0.9
    alpha[mask] = 0.0
    layer = np.zeros_like(img)
    layer[:] = INK_RGB
    return _blend(img, layer, alpha), alpha > 0.05


def hair(
    image: np.ndarray, rng: np.random.Generator, lesion: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Entre 5 y 10 curvas oscuras y delgadas (Bézier cuadráticas) cruzando la imagen."""
    img = _check(image)
    h, w = img.shape[:2]
    n = int(rng.integers(5, 11))
    stroke = np.zeros((h, w), dtype=np.uint8)
    t = np.linspace(0.0, 1.0, 64)[:, None]
    for _ in range(n):
        p0 = rng.uniform([0, 0], [w, h])
        p2 = rng.uniform([0, 0], [w, h])
        # Control desplazado del segmento para que la curva se arquee.
        mid = (p0 + p2) / 2 + rng.normal(0, 0.25 * min(h, w), size=2)
        pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * mid + t**2 * p2
        thickness = int(rng.integers(1, 3))
        cv2.polylines(stroke, [np.rint(pts).astype(np.int32)], False, 255, thickness, cv2.LINE_AA)
    alpha = stroke.astype(np.float32) / 255.0 * 0.9
    layer = np.zeros_like(img)
    layer[:] = HAIR_RGB
    return _blend(img, layer, alpha), alpha > 0.05


def vignette(
    image: np.ndarray, rng: np.random.Generator, lesion: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Oscurecimiento radial de las esquinas: el centro (``r < 0.45·r_max``) queda intacto y
    de ahí hacia afuera el factor cae cuadráticamente hasta ``1 − s`` en la esquina."""
    img = _check(image)
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(
        ((ys - (h - 1) / 2) ** 2 + (xs - (w - 1) / 2) ** 2)
        / (((h - 1) / 2) ** 2 + ((w - 1) / 2) ** 2)
    )
    ramp = np.clip((r - VIGNETTE_INNER) / (1.0 - VIGNETTE_INNER), 0.0, 1.0)
    factor = 1.0 - VIGNETTE_STRENGTH * ramp**2
    out = np.clip(np.rint(img.astype(np.float32) * factor[..., None]), 0, 255).astype(np.uint8)
    return out, factor < VIGNETTE_REGION_FACTOR


def noise(
    image: np.ndarray, rng: np.random.Generator, lesion: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Control: ruido gaussiano i.i.d. por píxel y canal, sin estructura espacial."""
    img = _check(image)
    delta = rng.normal(0.0, NOISE_SIGMA, size=img.shape).astype(np.float32)
    out = np.clip(np.rint(img.astype(np.float32) + delta), 0, 255).astype(np.uint8)
    return out, np.ones(img.shape[:2], dtype=bool)


PERTURBATIONS: dict[str, Perturbation] = {
    "ruler": ruler,
    "ink": ink,
    "hair": hair,
    "vignette": vignette,
    "noise": noise,
}

LABELS_ES = {
    "ruler": "regla",
    "ink": "tinta",
    "hair": "vello",
    "vignette": "viñeteado",
    "noise": "ruido (control)",
}


def pixel_change(original: np.ndarray, perturbed: np.ndarray) -> float:
    """Cambio medio absoluto por píxel y canal en escala 0–255 (magnitud de la perturbación)."""
    return float(np.abs(original.astype(np.float32) - perturbed.astype(np.float32)).mean())
