"""Datos: manifiesto, deduplicación, splits agrupados por paciente y redimensionado (F1).

El DataModule de Lightning llega en F2. La normalización de píxeles NO se define aquí:
proviene exclusivamente del diccionario que devuelve `melanoma.models.factory.build_model`.
"""

from melanoma.data.manifest import MANIFEST_COLUMNS, read_manifest

__all__ = ["MANIFEST_COLUMNS", "read_manifest"]
