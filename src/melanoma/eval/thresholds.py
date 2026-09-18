"""Umbral de operación por bootstrap a nivel paciente (F4.2).

Con 88 melanomas en validación, "sensibilidad ≥ 0.95" depende de cuatro casos: el umbral de
un solo corte es frágil. Se remuestrean pacientes con reemplazo, en cada remuestreo se calcula
el umbral que alcanza la sensibilidad objetivo y el umbral operativo es la mediana de esa
distribución (se reportan también los percentiles 5 y 95).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from melanoma.eval.metrics import confusion_at_threshold, threshold_for_sensitivity


def bootstrap_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    patient_ids: Sequence[str] | np.ndarray,
    sensitivity: float,
    n_resamples: int,
    seed: int,
) -> dict:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    pid = np.asarray(patient_ids).astype(str)
    patients, inverse = np.unique(pid, return_inverse=True)
    groups = [np.flatnonzero(inverse == k) for k in range(patients.size)]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        idx = np.concatenate([groups[k] for k in rng.integers(0, patients.size, patients.size)])
        if y[idx].sum() == 0:
            continue
        draws.append(threshold_for_sensitivity(p[idx], y[idx], sensitivity))
    draws_arr = np.asarray(draws)
    tau = float(np.median(draws_arr))
    return {
        "sensitivity_target": sensitivity,
        "tau": tau,
        "p05": float(np.percentile(draws_arr, 5)),
        "p95": float(np.percentile(draws_arr, 95)),
        "single_cut": threshold_for_sensitivity(p, y, sensitivity),
        "n_resamples": int(draws_arr.size),
        "unit": "patient",
        "seed": seed,
        "on_full_set": confusion_at_threshold(p, y, tau),
    }


def ppv_npv_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> dict:
    """VPP y VPN proyectados a otra prevalencia por aritmética directa (F4.7, sección 5)."""
    tp = sensitivity * prevalence
    fp = (1 - specificity) * (1 - prevalence)
    tn = specificity * (1 - prevalence)
    fn = (1 - sensitivity) * prevalence
    return {
        "prevalence": prevalence,
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
        "referred_fraction": tp + fp,
    }
