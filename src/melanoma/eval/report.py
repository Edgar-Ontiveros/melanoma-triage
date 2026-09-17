"""Tablas Markdown para los reportes de evaluación (F2.3)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _fmt(x: float, digits: int = 3) -> str:
    return "n/d" if x != x else f"{x:.{digits}f}"


def metrics_table(metrics: Mapping[str, Any], ci: Mapping[str, Any] | None = None) -> str:
    """Tabla ``métrica | valor | IC 95 %`` con las métricas de F2.3."""
    rows = [("AUC-ROC", "auroc"), ("AUPRC (primaria)", "auprc")]
    for key in sorted(k for k in metrics if k.startswith("sens_at_spec")):
        rows.append((f"Sensibilidad a especificidad {key[-3:-2]}.{key[-2:]}", key))
    for key in sorted(k for k in metrics if k.startswith("spec_at_sens")):
        rows.append((f"Especificidad a sensibilidad {key[-3:-2]}.{key[-2:]}", key))
    lines = ["| métrica | valor | IC 95 % (bootstrap por paciente) |", "|:--|--:|:--|"]
    for label, key in rows:
        interval = ""
        if ci and key in ci:
            interval = f"[{_fmt(ci[key]['lo'])}, {_fmt(ci[key]['hi'])}]"
        lines.append(f"| {label} | {_fmt(metrics[key])} | {interval} |")
    lines.append(
        f"| Prevalencia (AUPRC de un clasificador aleatorio) | {_fmt(metrics['prevalence'], 4)} | |"
    )
    return "\n".join(lines)


def confusion_table(metrics: Mapping[str, Any]) -> str:
    c = metrics["confusion"]
    return "\n".join(
        [
            f"Umbral: {c['threshold']:.4f} (política: {metrics['threshold_policy']}; la selección "
            "definitiva de umbral es de F4).",
            "",
            "| | pred. benigno | pred. melanoma |",
            "|:--|--:|--:|",
            f"| real benigno | {c['tn']} | {c['fp']} |",
            f"| real melanoma | {c['fn']} | {c['tp']} |",
            "",
            "| métrica en el umbral | valor |",
            "|:--|--:|",
            f"| Sensibilidad | {_fmt(c['sensitivity'])} |",
            f"| Especificidad | {_fmt(c['specificity'])} |",
            f"| VPP a la prevalencia observada ({metrics['prevalence']:.4f}) | {_fmt(c['ppv'])} |",
            f"| VPN a la prevalencia observada | {_fmt(c['npv'])} |",
        ]
    )
