"""Redimensionado: lado largo a ``long_side`` px, relación de aspecto intacta, JPEG ``quality``.

Nunca se agranda: una imagen cuyo lado largo ya es menor que ``long_side`` se reencoda tal
cual. El resultado se registra en el manifiesto como ``sha256_resized`` y ``relpath_resized``.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from melanoma.data.hashing import sha256_file


def resize_one(src: Path | str, dst: Path | str, long_side: int, quality: int) -> dict:
    """Redimensiona un archivo y devuelve ``sha256, width, height, bytes`` del resultado."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.draft("RGB", (long_side, long_side))  # decodificación reducida, nunca < long_side
        img = img.convert("RGB")
        w, h = img.size
        scale = long_side / max(w, h)
        if scale < 1.0:
            img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
        img.save(dst, format="JPEG", quality=quality)
        w, h = img.size
    return {"sha256": sha256_file(dst), "width": w, "height": h, "bytes": dst.stat().st_size}


def _worker(args: tuple[str, str, int, int]) -> dict:
    return resize_one(*args)


def resize_many(
    pairs: Sequence[tuple[Path | str, Path | str]], long_side: int, quality: int, workers: int
) -> list[dict]:
    """``resize_one`` sobre muchos (origen, destino) en procesos paralelos, en orden."""
    jobs = [(str(s), str(d), long_side, quality) for s, d in pairs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_worker, jobs, chunksize=32))
