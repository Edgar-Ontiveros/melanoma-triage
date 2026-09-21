"""Intervalos por bootstrap a nivel paciente para estadísticos de una variable por imagen
(solapamiento, Δ de probabilidad). Misma regla que ``melanoma.eval.bootstrap``: se remuestrean
pacientes con reemplazo y se toman todas sus imágenes, porque las imágenes de un mismo
paciente no son independientes."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

Statistic = Callable[[np.ndarray], float]


def bootstrap_cluster(
    values: np.ndarray,
    patient_ids: Sequence[str] | np.ndarray,
    n_resamples: int,
    seed: int,
    statistic: Statistic = np.mean,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """``{point, lo, hi, n, n_patients, n_resamples}`` con IC percentil ``1 − alpha``.

    Los ``nan`` de ``values`` se descartan antes (con sus pacientes); si no queda nada, el
    punto y los límites son ``nan``.
    """
    v = np.asarray(values, dtype=np.float64)
    pid = np.asarray(patient_ids).astype(str)
    if v.shape != pid.shape:
        raise ValueError(f"values {v.shape} y patient_ids {pid.shape} difieren")
    keep = ~np.isnan(v)
    v, pid = v[keep], pid[keep]
    out: dict[str, float | int] = {
        "n": int(v.size),
        "n_patients": int(len(np.unique(pid))),
        "n_resamples": int(n_resamples),
    }
    if v.size == 0:
        out.update(point=float("nan"), lo=float("nan"), hi=float("nan"))
        return out
    uniq, inverse = np.unique(pid, return_inverse=True)
    groups = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples)
    for b in range(n_resamples):
        draw = rng.integers(len(groups), size=len(groups))
        idx = np.concatenate([groups[g] for g in draw])
        stats[b] = statistic(v[idx])
    out.update(
        point=float(statistic(v)),
        lo=float(np.quantile(stats, alpha / 2)),
        hi=float(np.quantile(stats, 1 - alpha / 2)),
    )
    return out


def describe(values: np.ndarray) -> dict[str, float | int]:
    """Resumen descriptivo ignorando ``nan``: n, media, mediana, p05, p25, p75, p95."""
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p05": float(np.quantile(v, 0.05)),
        "p25": float(np.quantile(v, 0.25)),
        "p75": float(np.quantile(v, 0.75)),
        "p95": float(np.quantile(v, 0.95)),
    }
