"""Evaluación (F2.3): AUROC, AUPRC, sensibilidad/especificidad a nivel fijo, VPP/VPN, matriz de
confusión, bootstrap por paciente y curvas a archivo.

Todo número reportado en la tesis debe salir de aquí, acompañado de la config resuelta por
Hydra de la corrida que lo produjo. La exactitud no existe en este módulo.
"""

from melanoma.eval.bootstrap import bootstrap_patient, default_metric_fns
from melanoma.eval.metrics import assert_no_accuracy, compute_metrics, threshold_for_sensitivity
from melanoma.eval.plots import plot_all, plot_pr, plot_reliability, plot_roc
from melanoma.eval.report import confusion_table, metrics_table

__all__ = [
    "assert_no_accuracy",
    "bootstrap_patient",
    "compute_metrics",
    "confusion_table",
    "default_metric_fns",
    "metrics_table",
    "plot_all",
    "plot_pr",
    "plot_reliability",
    "plot_roc",
    "threshold_for_sensitivity",
]
