"""Resolución de rutas según el entorno de ejecución (F2.1).

Las rutas de imágenes, manifiesto y splits cambian entre la laptop y Kaggle. Ambas
variantes viven en ``configs/data/*.yaml`` (``data.paths.local`` y ``data.paths.kaggle``);
aquí solo se elige cuál aplica. ``data.env: auto`` detecta Kaggle por la variable de entorno
``KAGGLE_KERNEL_RUN_TYPE`` o por la existencia de ``/kaggle/input``. Nunca se edita código
para cambiar de entorno.
"""

from __future__ import annotations

import os
import re
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


_KAGGLE_DATASETS = re.compile(r"^/kaggle/input/datasets/[^/]+/([^/]+)(/.*)?$")


def kaggle_layout_candidates(path: Path | str) -> list[Path]:
    """Las dos estructuras con las que Kaggle monta un dataset, en orden de preferencia.

    Desde 2026-09 Kaggle usa ``/kaggle/input/datasets/<usuario>/<slug>/…``, pero algunas
    máquinas siguen montando la forma plana ``/kaggle/input/<slug>/…`` (el 2026-09-17, dos de
    seis kernels del mismo lote). Se aceptan ambas.
    """
    path = Path(path)
    m = _KAGGLE_DATASETS.match(path.as_posix())
    if not m:
        return [path]
    slug, rest = m.group(1), m.group(2) or ""
    return [path, Path(f"/kaggle/input/{slug}{rest}")]


def first_existing(path: Path | str) -> Path:
    """La primera variante de ``kaggle_layout_candidates`` que existe; si ninguna, la original."""
    candidates = kaggle_layout_candidates(path)
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


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
        path = path if path.is_absolute() else base / path
        return first_existing(path) if env == "kaggle" else path

    return DataPaths(
        env=env,
        images_dir=_abs(block["images_dir"]),
        manifest_path=_abs(block["manifest_path"]),
        splits_dir=_abs(block["splits_dir"]),
    )
