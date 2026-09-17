"""Entrenamiento: LightningModule binario, scheduler, callbacks y detector de colapso.

F0 dejó `LitBinaryClassifier` mínimo; F2.2 lo extiende con `pos_weight`, métricas de
validación (AUROC, AUPRC), cosine annealing con calentamiento y detección de colapso.
"""

from melanoma.train.callbacks import build_callbacks
from melanoma.train.collapse import CollapseReport, detect_collapse
from melanoma.train.module import LitBinaryClassifier
from melanoma.train.schedulers import cosine_with_warmup, cosine_with_warmup_factor

__all__ = [
    "CollapseReport",
    "LitBinaryClassifier",
    "build_callbacks",
    "cosine_with_warmup",
    "cosine_with_warmup_factor",
    "detect_collapse",
]
