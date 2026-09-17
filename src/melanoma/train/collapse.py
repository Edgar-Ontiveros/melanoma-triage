"""Detector de colapso (F2.2).

El modo de falla más probable con 1.8 % de positivos: el modelo predice siempre "benigno"
(o siempre lo mismo) y la pérdida se ve razonable. Se detecta al final de cada validación
con dos criterios, cualquiera basta:

1. la desviación estándar de las probabilidades predichas es menor que ``std_threshold``;
2. el AUROC dista menos de ``auc_tolerance`` de 0.5 (o no es calculable).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CollapseReport:
    collapsed: bool
    reason: str
    prob_std: float
    prob_mean: float
    auroc: float

    def as_dict(self) -> dict[str, float | bool | str]:
        return asdict(self)


def detect_collapse(
    probs: np.ndarray, auroc: float | None, std_threshold: float, auc_tolerance: float
) -> CollapseReport:
    """Evalúa los dos criterios sobre las probabilidades de una época de validación."""
    p = np.asarray(probs, dtype=np.float64).ravel()
    std = float(p.std()) if p.size else 0.0
    mean = float(p.mean()) if p.size else float("nan")
    auc = float("nan") if auroc is None else float(auroc)
    reasons = []
    if std < std_threshold:
        reasons.append(f"std de probabilidades {std:.5f} < {std_threshold}")
    if math.isnan(auc) or abs(auc - 0.5) < auc_tolerance:
        reasons.append(f"AUROC {auc:.4f} ≈ 0.5 (tolerancia {auc_tolerance})")
    return CollapseReport(
        collapsed=bool(reasons),
        reason="; ".join(reasons),
        prob_std=std,
        prob_mean=mean,
        auroc=auc,
    )
