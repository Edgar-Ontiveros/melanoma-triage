"""Huellas SHA256 de archivos y propiedades básicas de imágenes.

Sin constantes de configuración: tamaños de muestra, rutas y número de workers vienen
de ``configs/data/isic2020.yaml``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

_CHUNK = 1 << 20  # tamaño de lectura por bloque (no es configuración del pipeline)


def sha256_file(path: Path | str) -> str:
    """SHA256 hexadecimal del archivo completo, leído por bloques."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def image_size(path: Path | str) -> tuple[int, int]:
    """``(width, height)`` leídos de la cabecera, sin decodificar el JPEG."""
    with Image.open(path) as img:
        return img.size


def file_fingerprint(path: Path | str) -> dict[str, int | str]:
    """SHA256, ancho, alto y bytes de un archivo de imagen."""
    p = Path(path)
    w, h = image_size(p)
    return {"sha256": sha256_file(p), "width": w, "height": h, "bytes": p.stat().st_size}


def fingerprint_many(paths: Iterable[Path | str], workers: int) -> list[dict[str, int | str]]:
    """``file_fingerprint`` sobre muchos archivos con hilos (I/O y hashing liberan el GIL)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(file_fingerprint, list(paths)))
