"""Prueba de cordura de Adebayo et al. (2018), F5.2: reinicializar pesos y medir cuánto cambia
el CAM. Un método que produce el mismo mapa con pesos entrenados y aleatorios no explica el
modelo, explica la imagen.

Ámbitos de reinicialización (``scope``), en el orden de la spec:

- ``head``: capa lineal + ``conv_head`` + ``bn2`` (todo lo que sigue al último bloque).
- ``head_last_block``: lo anterior + ``blocks[-1]`` (última etapa de EfficientNetV2-S).
- ``all``: todos los parámetros.

La reinicialización usa ``reset_parameters`` de cada módulo (la inicialización por defecto de
torch: Kaiming uniforme en conv/lineal; BatchNorm a peso 1, sesgo 0 y estadísticas 0/1). La
métrica es la correlación de Spearman entre el CAM entrenado y el aleatorio, celda a celda
sobre la malla ``H × W``; el umbral de «depende del modelo» (< 0.3) está fijado en la spec.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from melanoma.explain.features import classifier_of

SCOPES = ("head", "head_last_block", "all")


def _modules_for_scope(model: nn.Module, scope: str) -> list[nn.Module]:
    clf = classifier_of(model)
    if scope == "head":
        return [clf.head, clf.backbone.conv_head, clf.backbone.bn2]
    if scope == "head_last_block":
        return [clf.head, clf.backbone.conv_head, clf.backbone.bn2, clf.backbone.blocks[-1]]
    if scope == "all":
        return [clf]
    raise ValueError(f"scope desconocido: {scope!r}; opciones: {SCOPES}")


def randomized_copy(model: nn.Module, scope: str, seed: int) -> nn.Module:
    """Copia profunda del modelo con los parámetros del ámbito reinicializados."""
    clone = copy.deepcopy(model)
    clone.eval()
    torch.manual_seed(seed)
    n_reset = 0
    for root in _modules_for_scope(clone, scope):
        for module in root.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
                n_reset += 1
    if n_reset == 0:
        raise RuntimeError(f"ningún módulo reinicializado para scope={scope!r}")
    return clone


def spearman_maps(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman entre dos mapas de la misma forma; ``nan`` si alguno es constante."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"mapas de forma distinta: {x.shape} vs {y.shape}")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def summarize_correlations(values: np.ndarray, threshold: float) -> dict:
    """Resumen de una lista de correlaciones por imagen; los ``nan`` (mapas constantes,
    típicamente un CAM aleatorio todo cero) se cuentan aparte y no entran en los promedios."""
    v = np.asarray(values, dtype=np.float64)
    ok = v[~np.isnan(v)]
    return {
        "n": int(v.size),
        "n_undefined": int(np.isnan(v).sum()),
        "mean": float(ok.mean()) if ok.size else None,
        "median": float(np.median(ok)) if ok.size else None,
        "p05": float(np.quantile(ok, 0.05)) if ok.size else None,
        "p95": float(np.quantile(ok, 0.95)) if ok.size else None,
        "mean_abs": float(np.abs(ok).mean()) if ok.size else None,
        "frac_below_threshold": float((ok < threshold).mean()) if ok.size else None,
        "threshold": float(threshold),
    }
