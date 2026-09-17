"""Datos: manifiesto, deduplicación, splits agrupados por paciente, redimensionado (F1) y
DataModule de Lightning (F2).

La normalización de píxeles NO se define aquí: proviene exclusivamente del diccionario que
devuelve `melanoma.models.factory.build_model` y se aplica en `melanoma.data.transforms`.
"""

from melanoma.data.datamodule import (
    ISICDataModule,
    LockedTestSplitError,
    datamodule_from_config,
    verify_split_hashes,
)
from melanoma.data.manifest import MANIFEST_COLUMNS, read_manifest
from melanoma.data.paths import DataPaths, detect_env, resolve_paths

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
