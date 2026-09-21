"""PNG del CAM superpuesto sobre el recorte que vio el modelo (no sobre la imagen original).
Solo numpy + Pillow; un mapa nulo nunca se superpone (la API devuelve ``png_base64: null`` y
solo el recorte)."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

ALPHA = 0.45


def _colormap(v: np.ndarray) -> np.ndarray:
    """Mapa azul → cian → verde → amarillo → rojo (tipo «jet» simplificado), ``v`` en [0, 1]."""
    v = np.clip(v, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay_png_base64(crop: np.ndarray, cam: np.ndarray | None, alpha: float = ALPHA) -> str:
    """``crop`` RGB uint8 [S, S, 3] y ``cam`` float [S, S] en [0, 1] → PNG en base64.
    Con ``cam=None`` devuelve el recorte tal cual (la vista «sin mapa» de la interfaz)."""
    if cam is None:
        out = np.ascontiguousarray(crop)
    else:
        if crop.shape[:2] != cam.shape:
            raise ValueError(f"crop {crop.shape[:2]} y cam {cam.shape} no coinciden")
        heat = _colormap(cam.astype(np.float32)) * 255.0
        blended = crop.astype(np.float32) * (1 - alpha) + heat * alpha
        out = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG", optimize=False, compress_level=3)
    return base64.b64encode(buf.getvalue()).decode("ascii")
