"""Entrenamiento: LightningModule binario y utilidades del bucle de entrenamiento.

En F0 solo existe `LitBinaryClassifier`, lo mínimo para que la prueba de humo
ejerza el camino completo config → modelo → optimizador → Trainer. Métricas,
callbacks y logging real llegan en F2.
"""

from melanoma.train.module import LitBinaryClassifier

__all__ = ["LitBinaryClassifier"]
