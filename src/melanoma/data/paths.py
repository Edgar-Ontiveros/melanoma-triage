"""Resolución de rutas según el entorno de ejecución (F2.1).

Las rutas de imágenes, manifiesto y splits cambian entre la laptop y Kaggle. Ambas
variantes viven en ``configs/data/*.yaml`` (``data.paths.local`` y ``data.paths.kaggle``);
aquí solo se elige cuál aplica. ``data.env: auto`` detecta Kaggle por la variable de entorno
``KAGGLE_KERNEL_RUN_TYPE`` o por la existencia de ``/kaggle/input``. Nunca se edita código
para cambiar de entorno.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ENVIRONMENTS = ("local", "kaggle")


@dataclass(frozen=True)
class DataPaths:
    env: str
    images_dir: Path
    manifest_path: Path
    splits_dir: Path


def detect_env(environ: Mapping[str, str] | None = None, kaggle_root: str = "/kaggle/input") -> str:
    """``kaggle`` si el proceso corre dentro de un kernel de Kaggle; ``local`` en otro caso."""
    environ = os.environ if environ is None else environ
    if environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path(kaggle_root).exists():
        return "kaggle"
    return "local"


def resolve_paths(data_cfg: Mapping, root: Path | str | None = None) -> DataPaths:
    """Elige el bloque de rutas de ``data_cfg.paths`` según ``data_cfg.env``.

    Args:
        data_cfg: el grupo ``data`` de la config (con ``env`` y ``paths``).
        root: directorio contra el que se resuelven rutas relativas (raíz del repo).
    """
    env = str(data_cfg["env"])
    if env == "auto":
        env = detect_env()
    if env not in ENVIRONMENTS:
        raise ValueError(
            f"data.env desconocido: {env!r}; opciones: auto, {', '.join(ENVIRONMENTS)}"
        )
    block = data_cfg["paths"][env]
    base = Path(root) if root is not None else Path.cwd()

    def _abs(p: str) -> Path:
        path = Path(str(p))
        return path if path.is_absolute() else base / path

    return DataPaths(
        env=env,
        images_dir=_abs(block["images_dir"]),
        manifest_path=_abs(block["manifest_path"]),
        splits_dir=_abs(block["splits_dir"]),
    )
