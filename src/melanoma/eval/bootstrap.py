"""Intervalos de confianza por bootstrap a nivel paciente (F2.3).

Se remuestrean PACIENTES con reemplazo y se toman todas sus imágenes. Las imágenes de un
mismo paciente no son independientes; remuestrear imágenes daría intervalos falsamente
estrechos.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_curve

from melanoma.eval.metrics import (
    _validate,
    auroc_auprc,
    sensitivity_at_specificity,
    specificity_at_sensitivity,
)

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def default_metric_fns(fixed_levels: Sequence[float] = (0.90, 0.95)) -> dict[str, MetricFn]:
    fns: dict[str, MetricFn] = {
        "auroc": lambda p, y: auroc_auprc(p, y)[0],
        "auprc": lambda p, y: auroc_auprc(p, y)[1],
    }
    for level in fixed_levels:
        tag = f"{level:.2f}".replace(".", "")

        def _sens(p: np.ndarray, y: np.ndarray, lv: float = level) -> float:
            fpr, tpr, _ = roc_curve(y, p)
            return sensitivity_at_specificity(fpr, tpr, lv)

        def _spec(p: np.ndarray, y: np.ndarray, lv: float = level) -> float:
            fpr, tpr, _ = roc_curve(y, p)
            return specificity_at_sensitivity(fpr, tpr, lv)

        fns[f"sens_at_spec{tag}"] = _sens
        fns[f"spec_at_sens{tag}"] = _spec
    return fns


def bootstrap_patient(
    probs: np.ndarray,
    labels: np.ndarray,
    patient_ids: Sequence[str] | np.ndarray,
    n_resamples: int,
    seed: int,
    metric_fns: Mapping[str, MetricFn] | None = None,
    alpha: float = 0.05,
) -> dict[str, dict[str, float | int]]:
    """``{métrica: {point, lo, hi, n_resamples, n_skipped}}`` con IC percentil (1-alpha).

    Los remuestreos sin positivos (o sin negativos) no permiten calcular AUROC/AUPRC y se
    descartan; se reporta cuántos fueron.
    """
    p, y = _validate(probs, labels)
    pid = np.asarray(patient_ids).astype(str).ravel()
    if pid.shape != p.shape:
        raise ValueError("patient_ids no tiene la longitud de probs")
    metric_fns = default_metric_fns() if metric_fns is None else dict(metric_fns)
    patients, inverse = np.unique(pid, return_inverse=True)
    groups = [np.flatnonzero(inverse == k) for k in range(patients.size)]
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in metric_fns}
    skipped = 0
    for _ in range(n_resamples):
        draw = rng.integers(0, patients.size, size=patients.size)
        idx = np.concatenate([groups[k] for k in draw])
        yb = y[idx]
        if yb.min() == yb.max():
            skipped += 1
            continue
        pb = p[idx]
        for name, fn in metric_fns.items():
            samples[name].append(fn(pb, yb))
    out: dict[str, dict[str, float | int]] = {}
    for name, fn in metric_fns.items():
        vals = np.asarray(samples[name], dtype=np.float64)
        lo, hi = (
            (
                float(np.nanpercentile(vals, 100 * alpha / 2)),
                float(np.nanpercentile(vals, 100 * (1 - alpha / 2))),
            )
            if vals.size
            else (float("nan"), float("nan"))
        )
        out[name] = {
            "point": float(fn(p, y)),
            "lo": lo,
            "hi": hi,
            "n_resamples": int(vals.size),
            "n_skipped": skipped,
            "level": 1 - alpha,
            "unit": "patient",
        }
    return out
