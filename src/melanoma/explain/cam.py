"""CAM en numpy puro (F5.1). Este es el contrato que consume la API de F6.

Con la cabeza del proyecto (pooling global promedio → dropout → lineal de una salida), el
gradiente del logit respecto a cada mapa de activación ``A_k(x, y)`` es constante e igual a
``w_k / (H·W)``. Los pesos de Grad-CAM (promedio espacial del gradiente) valen entonces
``w_k / (H·W)`` y, como Grad-CAM normaliza el mapa al máximo, el factor ``1 / (H·W)`` se
cancela:

    Grad-CAM(x, y) = ReLU( Σ_k w_k · A_k(x, y) )  =  CAM de Zhou et al. (2016)

Aquí no hay torch: solo los mapas ``A_k`` (salida de la última etapa convolucional, para
EfficientNetV2-S a 224 px un tensor ``1280 × 7 × 7``) y los pesos ``w_k`` de la capa lineal.
La verificación numérica contra ``pytorch-grad-cam`` vive en ``tests/test_explain.py`` y en
``scripts/f5_cam.py``.

Contrato (F6):

    explain(features[C, H, W], linear_weight[C]) -> cam[H, W]   valores en [0, 1], máximo 1
    upsample(cam[H, W], size) -> [size, size]                    interpolación bilineal
"""

from __future__ import annotations

import numpy as np


def cam_from_features(features: np.ndarray, linear_weight: np.ndarray) -> np.ndarray:
    """Mapa de activación de clase para una imagen o un lote.

    Args:
        features: ``[C, H, W]`` o ``[B, C, H, W]``, salida de la última etapa convolucional
            (lo que entra al pooling global).
        linear_weight: ``[C]`` (o ``[1, C]``), pesos de la capa lineal de una salida.

    Returns:
        ``[H, W]`` (o ``[B, H, W]``) en ``float32``: ``ReLU(Σ_k w_k A_k)`` dividido entre su
        máximo, de modo que el máximo vale 1. Si ningún píxel es positivo el mapa es cero
        (no hay evidencia a favor de la clase y se devuelve tal cual, sin dividir entre 0).
    """
    a = np.asarray(features, dtype=np.float32)
    w = np.asarray(linear_weight, dtype=np.float32).reshape(-1)
    squeeze = a.ndim == 3
    if squeeze:
        a = a[None]
    if a.ndim != 4:
        raise ValueError(f"features debe ser [C,H,W] o [B,C,H,W]; llegó {a.shape}")
    if a.shape[1] != w.shape[0]:
        raise ValueError(f"canales {a.shape[1]} ≠ pesos {w.shape[0]}")
    cam = np.einsum("bchw,c->bhw", a, w)
    cam = np.maximum(cam, 0.0)
    peak = cam.reshape(cam.shape[0], -1).max(axis=1)
    scale = np.where(peak > 0, peak, 1.0)
    cam = cam / scale[:, None, None]
    return cam[0] if squeeze else cam


def upsample(cam: np.ndarray, size: int) -> np.ndarray:
    """Interpolación bilineal ``[H, W] → [size, size]`` en numpy puro.

    Convención de centros de píxel desplazados medio píxel (la de ``torch`` con
    ``align_corners=False`` y la de OpenCV ``INTER_LINEAR``), con las coordenadas fuente
    recortadas al borde. ``tests/test_explain.py`` la compara con ``torch``.
    """
    c = np.asarray(cam, dtype=np.float32)
    if c.ndim != 2:
        raise ValueError(f"cam debe ser [H, W]; llegó {c.shape}")
    size = int(size)
    if size <= 0:
        raise ValueError("size debe ser positivo")
    rows = _bilinear_axis(c.shape[0], size)
    cols = _bilinear_axis(c.shape[1], size)
    r0, r1, rw = rows
    c0, c1, cw = cols
    top = c[r0][:, c0] * (1 - cw)[None, :] + c[r0][:, c1] * cw[None, :]
    bottom = c[r1][:, c0] * (1 - cw)[None, :] + c[r1][:, c1] * cw[None, :]
    return (top * (1 - rw)[:, None] + bottom * rw[:, None]).astype(np.float32)


def _bilinear_axis(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Índices ``(i0, i1)`` y peso de ``i1`` para cada posición de salida en un eje."""
    scale = n_in / n_out
    src = (np.arange(n_out, dtype=np.float64) + 0.5) * scale - 0.5
    src = np.clip(src, 0.0, n_in - 1)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_in - 1)
    return i0, i1, (src - i0).astype(np.float32)


def energy_fraction(cam: np.ndarray, region: np.ndarray) -> float:
    """Fracción de la energía del mapa (suma de valores) que cae dentro de ``region``.

    ``cam`` y ``region`` deben tener la misma forma; ``region`` es booleana. Si el mapa es
    cero en todas partes la fracción no está definida y se devuelve ``nan``.
    """
    c = np.asarray(cam, dtype=np.float64)
    m = np.asarray(region, dtype=bool)
    if c.shape != m.shape:
        raise ValueError(f"cam {c.shape} y region {m.shape} difieren")
    total = c.sum()
    if total <= 0:
        return float("nan")
    return float(c[m].sum() / total)


# Nombre del contrato de F6: la API llama ``explain`` y no necesita saber de Grad-CAM.
explain = cam_from_features
