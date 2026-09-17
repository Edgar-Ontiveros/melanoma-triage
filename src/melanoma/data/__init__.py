"""Datos: manifiesto, deduplicación, splits agrupados por paciente, redimensionado (F1) y
DataModule de Lightning (F2).

Los módulos de F1 (`manifest`, `hashing`, `dedup`, `splits`, `resize`, `paths`) solo necesitan
el grupo `core` (pandas, Pillow, numpy). El DataModule y las transformaciones necesitan el grupo
`train` (torch, lightning, albumentations) y se exponen de forma perezosa: importar
`melanoma.data` o `melanoma.data.resize` no debe cargar torch. Lo vigila
`tests/test_notebooks.py::test_core_modules_import_without_torch`.

La normalización de píxeles NO se define aquí: proviene exclusivamente del diccionario que
devuelve `melanoma.models.factory.build_model` y se aplica en `melanoma.data.transforms`.
"""

from __future__ import annotations

from typing import Any

from melanoma.data.manifest import MANIFEST_COLUMNS, read_manifest
from melanoma.data.paths import DataPaths, detect_env, resolve_paths

_TRAIN_ONLY = {
    "ISICDataModule": "melanoma.data.datamodule",
    "LockedTestSplitError": "melanoma.data.datamodule",
    "datamodule_from_config": "melanoma.data.datamodule",
    "verify_split_hashes": "melanoma.data.datamodule",
}

__all__ = [
    "MANIFEST_COLUMNS",
    "DataPaths",
    "ISICDataModule",
    "LockedTestSplitError",
    "datamodule_from_config",
    "detect_env",
    "read_manifest",
    "resolve_paths",
    "verify_split_hashes",
]


def __getattr__(name: str) -> Any:
    module = _TRAIN_ONLY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)
