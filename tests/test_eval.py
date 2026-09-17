"""Evaluación (F2.3): métricas correctas en casos conocidos, bootstrap por paciente, gráficas a
archivo y ausencia total de exactitud."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from melanoma.eval import (
    assert_no_accuracy,
    bootstrap_patient,
    compute_metrics,
    confusion_table,
    metrics_table,
    plot_all,
    threshold_for_sensitivity,
)


def _data(seed: int = 0, n: int = 2000, per_patient: int = 5):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.03).astype(int)
    p = np.clip(rng.normal(0.15 + 0.5 * y, 0.2), 0, 1)
    pid = np.array([f"P{i // per_patient}" for i in range(n)])
    return p, y, pid


def test_perfect_and_random_classifiers() -> None:
    y = np.array([0] * 90 + [1] * 10)
    perfect = compute_metrics(y.astype(float), y, np.arange(100).astype(str))
    assert perfect["auroc"] == 1.0 and perfect["auprc"] == 1.0
    assert perfect["sens_at_spec095"] == 1.0 and perfect["spec_at_sens095"] == 1.0
    assert perfect["prevalence"] == pytest.approx(0.1)
    inverted = compute_metrics(1.0 - y, y, np.arange(100).astype(str))
    assert inverted["auroc"] == 0.0
    assert inverted["auprc"] == pytest.approx(perfect["auprc_random"], abs=0.02)


def test_metrics_contain_everything_and_no_accuracy() -> None:
    p, y, pid = _data()
    m = compute_metrics(p, y, pid)
    for key in (
        "auroc",
        "auprc",
        "prevalence",
        "auprc_random",
        "n_patients",
        "sens_at_spec090",
        "sens_at_spec095",
        "spec_at_sens090",
        "spec_at_sens095",
    ):
        assert key in m
    for key in ("tn", "fp", "fn", "tp", "sensitivity", "specificity", "ppv", "npv"):
        assert key in m["confusion"]
    assert set(m["curves"]) == {"roc", "pr", "reliability"}
    assert m["n_patients"] == 400
    assert_no_accuracy(m)
    text = metrics_table(m) + confusion_table(m)
    assert "accuracy" not in text.lower() and "exactitud" not in text.lower()
    with pytest.raises(AssertionError):
        assert_no_accuracy({"metrics": {"accuracy": 0.98}})


def test_threshold_policy_reaches_target_sensitivity() -> None:
    p, y, pid = _data()
    thr = threshold_for_sensitivity(p, y, 0.90)
    m = compute_metrics(p, y, pid, threshold_sensitivity=0.90)
    assert m["confusion"]["threshold"] == pytest.approx(thr)
    assert m["confusion"]["sensitivity"] >= 0.90
    fixed = compute_metrics(p, y, pid, threshold=0.5)
    assert fixed["threshold_policy"] == "fijo" and fixed["confusion"]["threshold"] == 0.5


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        compute_metrics(np.array([1.2, 0.1]), np.array([1, 0]), ["a", "b"])
    with pytest.raises(ValueError):
        compute_metrics(np.array([0.2, 0.1]), np.array([1, 2]), ["a", "b"])
    with pytest.raises(ValueError):
        compute_metrics(np.array([0.2, 0.1]), np.array([1, 0]), ["a"])


def test_bootstrap_is_patient_level() -> None:
    """Con etiquetas idénticas dentro de cada paciente, el IC por paciente debe ser más ancho que
    el IC por imagen (las imágenes de un paciente no son independientes)."""
    rng = np.random.default_rng(1)
    n_pat, per = 150, 12
    y_pat = (rng.random(n_pat) < 0.2).astype(int)
    y = np.repeat(y_pat, per)
    p = np.clip(
        np.repeat(rng.normal(0.2 + 0.4 * y_pat, 0.15), per) + rng.normal(0, 0.02, y.size), 0, 1
    )
    pid = np.repeat([f"P{i}" for i in range(n_pat)], per)
    by_patient = bootstrap_patient(p, y, pid, n_resamples=300, seed=0)
    by_image = bootstrap_patient(p, y, np.arange(y.size).astype(str), n_resamples=300, seed=0)
    for k in ("auroc", "auprc"):
        assert by_patient[k]["lo"] <= by_patient[k]["point"] <= by_patient[k]["hi"]
        assert by_patient[k]["unit"] == "patient"
    width_pat = by_patient["auroc"]["hi"] - by_patient["auroc"]["lo"]
    width_img = by_image["auroc"]["hi"] - by_image["auroc"]["lo"]
    assert width_pat > width_img


def test_bootstrap_deterministic_and_reports_skips() -> None:
    p, y, pid = _data(n=300)
    a = bootstrap_patient(p, y, pid, n_resamples=50, seed=3)
    b = bootstrap_patient(p, y, pid, n_resamples=50, seed=3)
    assert a == b
    assert a["auroc"]["n_resamples"] + a["auroc"]["n_skipped"] == 50


def test_plots_written_to_file(tmp_path: Path) -> None:
    p, y, pid = _data(n=500)
    m = compute_metrics(p, y, pid)
    paths = plot_all(m, tmp_path, "unit")
    assert set(paths) == {"roc", "pr", "reliability"}
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 1000
