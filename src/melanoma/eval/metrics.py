"""Métricas de clasificación binaria a baja prevalencia (F2.3).

Contrato: probabilidades, etiquetas y ``patient_id`` → diccionario con métricas y los datos
de las curvas. Métrica primaria: AUPRC (a 1.77 % de prevalencia el AUC-ROC es engañoso y la
exactitud, inútil). **La exactitud no se calcula ni se reporta**: un modelo que predice
"todo benigno" obtiene 98.2 %.

F4 reutiliza este módulo tal cual sobre el conjunto de prueba.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

FORBIDDEN_METRICS = ("accuracy", "acc")


def _validate(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(labels).astype(int).ravel()
    if p.shape != y.shape:
        raise ValueError(f"probs {p.shape} y labels {y.shape} no tienen la misma longitud")
    if p.size == 0:
        raise ValueError("no hay predicciones")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("las probabilidades deben estar en [0, 1]")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("las etiquetas deben ser 0/1")
    return p, y


def auroc_auprc(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """``(AUROC, AUPRC)``; ``nan`` si solo hay una clase."""
    p, y = _validate(probs, labels)
    if y.min() == y.max():
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def sensitivity_at_specificity(fpr: np.ndarray, tpr: np.ndarray, spec: float) -> float:
    """Mayor sensibilidad alcanzable con especificidad ≥ ``spec`` (de la curva ROC)."""
    ok = (1.0 - fpr) >= spec
    return float(tpr[ok].max()) if ok.any() else 0.0


def specificity_at_sensitivity(fpr: np.ndarray, tpr: np.ndarray, sens: float) -> float:
    """Mayor especificidad alcanzable con sensibilidad ≥ ``sens`` (de la curva ROC)."""
    ok = tpr >= sens
    return float((1.0 - fpr[ok]).max()) if ok.any() else 0.0


def threshold_for_sensitivity(probs: np.ndarray, labels: np.ndarray, sens: float) -> float:
    """Umbral más alto cuya sensibilidad es ≥ ``sens``."""
    p, y = _validate(probs, labels)
    pos = np.sort(p[y == 1])
    if pos.size == 0:
        return 0.5
    # Con umbral t, sensibilidad = fracción de positivos con p >= t.
    k = int(np.ceil(sens * pos.size))
    k = min(max(k, 1), pos.size)
    return float(pos[pos.size - k])


def confusion_at_threshold(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    p, y = _validate(probs, labels)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    return {
        "threshold": float(threshold),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
    }


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> dict[str, list]:
    """Diagrama de fiabilidad: por bin de probabilidad, confianza media y fracción observada."""
    p, y = _validate(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    conf, obs, count = [], [], []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        count.append(n)
        conf.append(float(p[mask].mean()) if n else float("nan"))
        obs.append(float(y[mask].mean()) if n else float("nan"))
    return {"edges": edges.tolist(), "confidence": conf, "observed": obs, "count": count}


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    patient_ids: Sequence[str] | np.ndarray,
    fixed_levels: Sequence[float] = (0.90, 0.95),
    threshold: float | None = None,
    threshold_sensitivity: float = 0.90,
    n_reliability_bins: int = 10,
) -> dict[str, Any]:
    """Todas las métricas de F2.3 y los datos para las gráficas.

    Args:
        probs: probabilidad de melanoma por imagen.
        labels: etiqueta 0/1 por imagen.
        patient_ids: paciente de cada imagen (para conteos; el bootstrap va aparte).
        fixed_levels: niveles de especificidad/sensibilidad fija a reportar.
        threshold: umbral para la matriz de confusión y VPP/VPN. Si es ``None`` se usa el
            umbral que alcanza ``threshold_sensitivity`` (provisional; la selección real es F4).
        n_reliability_bins: bins del diagrama de fiabilidad.
    """
    p, y = _validate(probs, labels)
    pid = np.asarray(patient_ids).astype(str).ravel()
    if pid.shape != p.shape:
        raise ValueError("patient_ids no tiene la longitud de probs")
    n_pos = int(y.sum())
    prevalence = n_pos / y.size
    auroc, auprc = auroc_auprc(p, y)
    fpr, tpr, roc_thr = (
        roc_curve(y, p)
        if n_pos and n_pos < y.size
        else (np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, 0.0]))
    )
    precision, recall, pr_thr = precision_recall_curve(y, p)
    thr = threshold_for_sensitivity(p, y, threshold_sensitivity) if threshold is None else threshold
    out: dict[str, Any] = {
        "n_images": int(y.size),
        "n_positives": n_pos,
        "n_patients": int(np.unique(pid).size),
        "prevalence": float(prevalence),
        "auprc_random": float(prevalence),
        "auroc": auroc,
        "auprc": auprc,
        "threshold_policy": (
            f"sensibilidad>={threshold_sensitivity}" if threshold is None else "fijo"
        ),
        "confusion": confusion_at_threshold(p, y, thr),
        "curves": {
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thr.tolist()},
            "pr": {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": pr_thr.tolist(),
            },
            "reliability": reliability_bins(p, y, n_reliability_bins),
        },
    }
    for level in fixed_levels:
        tag = f"{level:.2f}".replace(".", "")
        out[f"sens_at_spec{tag}"] = sensitivity_at_specificity(fpr, tpr, level)
        out[f"spec_at_sens{tag}"] = specificity_at_sensitivity(fpr, tpr, level)
    for key in FORBIDDEN_METRICS:
        assert key not in out, "la exactitud no se reporta en este proyecto"
    return out


def assert_no_accuracy(obj: Any, path: str = "") -> None:
    """Recorre un diccionario de métricas y falla si aparece cualquier clave de exactitud."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in FORBIDDEN_METRICS or "accuracy" in str(k).lower():
                raise AssertionError(f"métrica prohibida en {path}/{k}")
            assert_no_accuracy(v, f"{path}/{k}")
    elif isinstance(obj, list | tuple):
        for i, v in enumerate(obj):
            assert_no_accuracy(v, f"{path}[{i}]")
