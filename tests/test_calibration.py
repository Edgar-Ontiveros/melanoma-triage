"""Calibración (F4.1) y umbral por bootstrap (F4.2): orden preservado, Platt recupera un
desplazamiento conocido, ECE/Brier razonables, proyección de VPP/VPN."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from melanoma.eval import (
    bootstrap_threshold,
    brier,
    calibration_summary,
    ece,
    fit_platt,
    fit_temperature,
    ppv_npv_at_prevalence,
)


def _synthetic(seed: int = 0, n: int = 4000, shift: float = 0.0, scale: float = 1.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    true_logit = rng.normal(-3.0 + 3.0 * y, 1.0)
    # el modelo "observa" logits desplazados y reescalados
    z = scale * true_logit + shift
    return z, y


def test_calibration_preserves_ranking() -> None:
    z, y = _synthetic(shift=4.0, scale=1.5)
    raw = 1 / (1 + np.exp(-z))
    for cal in (fit_platt(z, y), fit_temperature(z, y)):
        p = cal.apply(z)
        assert roc_auc_score(y, p) == pytest.approx(roc_auc_score(y, raw), abs=1e-12)
        assert average_precision_score(y, p) == pytest.approx(
            average_precision_score(y, raw), abs=1e-12
        )


def test_platt_recovers_known_shift() -> None:
    """Logits bien calibrados desplazados +4 (≈ ln 55): Platt debe devolver b ≈ −4, a ≈ 1."""
    rng = np.random.default_rng(1)
    n = 20000
    true_logit = rng.normal(-3.0, 2.0, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-true_logit))).astype(int)
    cal = fit_platt(true_logit + 4.0, y)
    assert cal.b == pytest.approx(-4.0, abs=0.25)
    assert cal.a == pytest.approx(1.0, abs=0.1)


def test_platt_improves_calibration_metrics() -> None:
    z, y = _synthetic(shift=4.0, scale=1.5)
    raw = 1 / (1 + np.exp(-z))
    cal = fit_platt(z, y).apply(z)
    assert brier(cal, y) < brier(raw, y)
    assert ece(cal, y)["ece"] < ece(raw, y)["ece"]
    s = calibration_summary(cal, y)
    assert set(s) == {"brier", "ece_uniform", "ece_quantile"}
    assert sum(b["n"] for b in s["ece_uniform"]["bins"]) == y.size
    with pytest.raises(ValueError):
        ece(cal, y, strategy="otra")


def test_bootstrap_threshold_is_median_with_percentiles() -> None:
    z, y = _synthetic()
    p = 1 / (1 + np.exp(-z))
    pid = np.array([f"P{i // 8}" for i in range(y.size)])
    res = bootstrap_threshold(p, y, pid, sensitivity=0.95, n_resamples=200, seed=0)
    assert res["p05"] <= res["tau"] <= res["p95"]
    assert res["unit"] == "patient" and res["n_resamples"] <= 200
    assert res["on_full_set"]["threshold"] == pytest.approx(res["tau"])
    assert res["on_full_set"]["sensitivity"] >= 0.85  # mediana de umbrales: cerca del objetivo


def test_ppv_npv_projection() -> None:
    r = ppv_npv_at_prevalence(sensitivity=0.95, specificity=0.30, prevalence=0.02)
    assert r["ppv"] == pytest.approx(0.95 * 0.02 / (0.95 * 0.02 + 0.70 * 0.98))
    assert r["npv"] == pytest.approx(0.30 * 0.98 / (0.30 * 0.98 + 0.05 * 0.02))
    assert 0 < r["referred_fraction"] < 1
