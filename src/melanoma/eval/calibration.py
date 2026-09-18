"""Calibración post hoc sobre validación (F4.1): Platt y temperature scaling, ECE y Brier.

Con ``pos_weight`` ≈ 55 el modelo aprendió como si los positivos fueran 55 veces más
frecuentes: los logits quedan desplazados ≈ +ln(55) además de la sobreconfianza habitual.
Temperature (``z / T``) corrige solo la escala; Platt (``a·z + b``) corrige escala y
desplazamiento, por eso es la primaria. Ambas son monótonas: no cambian AUROC ni AUPRC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, log_expit


def _nll(logits: np.ndarray, y: np.ndarray) -> float:
    """Entropía cruzada binaria media a partir de logits (estable numéricamente)."""
    return float(-np.mean(y * log_expit(logits) + (1 - y) * log_expit(-logits)))


@dataclass(frozen=True)
class Platt:
    a: float
    b: float

    def apply(self, logits: np.ndarray) -> np.ndarray:
        return expit(self.a * np.asarray(logits, dtype=np.float64) + self.b)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class Temperature:
    T: float

    def apply(self, logits: np.ndarray) -> np.ndarray:
        return expit(np.asarray(logits, dtype=np.float64) / self.T)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> Platt:
    """Minimiza la entropía cruzada de ``sigmoid(a·z + b)``; sin regularización."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    res = minimize(lambda p: _nll(p[0] * z + p[1], y), x0=np.array([1.0, 0.0]), method="BFGS")
    return Platt(a=float(res.x[0]), b=float(res.x[1]))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> Temperature:
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    res = minimize_scalar(lambda t: _nll(z / t, y), bounds=(0.05, 100.0), method="bounded")
    return Temperature(T=float(res.x))


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10, strategy: str = "uniform") -> dict:
    """Error de calibración esperado, ponderado por el tamaño de cada bin.

    ``strategy``: ``uniform`` (bins de ancho fijo en [0, 1]) o ``quantile`` (bins con el mismo
    número de muestras). Con 1.77 % de prevalencia casi todo cae en el primer bin uniforme, por
    eso se reportan los dos y el conteo por bin.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    else:
        raise ValueError(f"strategy desconocida: {strategy!r}")
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    total, rows = 0.0, []
    for b in range(len(edges) - 1):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": b, "n": 0, "confidence": None, "observed": None})
            continue
        conf, obs = float(p[mask].mean()), float(y[mask].mean())
        total += n / p.size * abs(conf - obs)
        rows.append({"bin": b, "n": n, "confidence": conf, "observed": obs})
    return {"ece": float(total), "strategy": strategy, "edges": edges.tolist(), "bins": rows}


def calibration_summary(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    return {
        "brier": brier(probs, labels),
        "ece_uniform": ece(probs, labels, n_bins, "uniform"),
        "ece_quantile": ece(probs, labels, n_bins, "quantile"),
    }
