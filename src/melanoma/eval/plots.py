"""Gráficas de evaluación a archivo (F2.3): ROC, precisión-recall y fiabilidad."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_roc(metrics: Mapping[str, Any], path: Path | str, title: str = "ROC") -> Path:
    roc = metrics["curves"]["roc"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(roc["fpr"], roc["tpr"], label=f"AUROC = {metrics['auroc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="azar")
    ax.set_xlabel("1 - especificidad (FPR)")
    ax.set_ylabel("sensibilidad (TPR)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_pr(metrics: Mapping[str, Any], path: Path | str, title: str = "Precisión-recall") -> Path:
    pr = metrics["curves"]["pr"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(pr["recall"], pr["precision"], label=f"AUPRC = {metrics['auprc']:.3f}")
    ax.axhline(
        metrics["prevalence"],
        linestyle="--",
        color="gray",
        label=f"azar = prevalencia {metrics['prevalence']:.4f}",
    )
    ax.set_xlabel("recall (sensibilidad)")
    ax.set_ylabel("precisión (VPP)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper right")
    return _save(fig, path)


def plot_reliability(
    metrics: Mapping[str, Any], path: Path | str, title: str = "Diagrama de fiabilidad"
) -> Path:
    rel = metrics["curves"]["reliability"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="calibración perfecta")
    ax.plot(rel["confidence"], rel["observed"], marker="o", label="modelo")
    for c, o, n in zip(rel["confidence"], rel["observed"], rel["count"], strict=True):
        if n:
            ax.annotate(str(n), (c, o), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("probabilidad predicha (media del bin)")
    ax.set_ylabel("fracción observada de melanoma")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left")
    return _save(fig, path)


def plot_all(metrics: Mapping[str, Any], out_dir: Path | str, prefix: str) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "roc": plot_roc(metrics, out_dir / f"{prefix}_roc.png", f"ROC — {prefix}"),
        "pr": plot_pr(metrics, out_dir / f"{prefix}_pr.png", f"Precisión-recall — {prefix}"),
        "reliability": plot_reliability(
            metrics, out_dir / f"{prefix}_reliability.png", f"Fiabilidad — {prefix}"
        ),
    }


def _save(fig: plt.Figure, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
