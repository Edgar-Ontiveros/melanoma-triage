# ruff: noqa: E501
"""F6.6 — Paquete de prueba para CI y Docker: el mismo backbone con pesos aleatorios (sin
descargar nada), preprocess.json real de timm y calibración/umbrales/métricas sintéticos y
marcados como tales. Misma estructura y tamaño que el paquete real.

Uso: uv run python scripts/make_test_bundle.py [--out models/test_bundle] [--negative-bias B]
``--negative-bias`` fija el sesgo de la capa lineal (p. ej. −50) para que toda imagen dé logit
muy negativo y ``--zero-weights`` anula los pesos de la capa lineal, con lo que el CAM es nulo
→ caso ``no_positive_evidence``. Se usa también en las pruebas (``make_test_bundle.build``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from melanoma.export.bundle import write_bundle
from melanoma.models import build_model

ROOT = Path(__file__).resolve().parents[1]
BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"
IMAGE_SIZE = 224
OPSET = 17

SYNTHETIC_CALIBRATION = {
    "method": "platt",
    "a": 0.5,
    "b": -4.0,
    "fitted_on": "synthetic",
    "n_images": 0,
    "n_positives": 0,
    "date": "n/a",
    "source": "make_test_bundle.py (SINTÉTICO, no usar para servir)",
}
SYNTHETIC_THRESHOLDS = {
    "operating": "tau_95",
    "source": "make_test_bundle.py (SINTÉTICO)",
    "tau_95": {
        "tau": 0.004,
        "sensitivity_target": 0.95,
        "p05": 0.001,
        "p95": 0.006,
        "validation": {"sensitivity": 0.95, "specificity": 0.4, "ppv": 0.03, "npv": 0.998},
    },
    "tau_90": {
        "tau": 0.007,
        "sensitivity_target": 0.90,
        "p05": 0.004,
        "p95": 0.011,
        "validation": {"sensitivity": 0.9, "specificity": 0.54, "ppv": 0.034, "npv": 0.997},
    },
}
SYNTHETIC_METRICS = {
    "split": "synthetic",
    "accessed_once": False,
    "date": "n/a",
    "n_images": 0,
    "n_positives": 0,
    "n_patients": 0,
    "auroc": {"point": 0.5, "lo": 0.4, "hi": 0.6},
    "auprc": {"point": 0.02, "lo": 0.01, "hi": 0.03},
    "prevalence": 0.017,
    "tau_95": {
        "threshold": 0.004,
        "sensitivity": 0.95,
        "specificity": 0.4,
        "ppv": 0.03,
        "npv": 0.998,
        "tp": 0,
        "fn": 0,
        "fp": 0,
        "tn": 0,
    },
    "ci": {"level": 0.95, "unit": "patient", "n_resamples": 0},
    "source": "make_test_bundle.py (SINTÉTICO)",
}


def build(
    out: Path,
    seed: int = 0,
    negative_bias: float | None = None,
    zero_weights: bool = False,
) -> dict:
    torch.manual_seed(seed)
    model, data_config = build_model(BACKBONE, pretrained=False, num_classes=1, dropout=0.2)
    with torch.no_grad():
        if zero_weights:  # CAM nulo garantizado: ReLU(Σ 0 · A_k) = 0 → «no_positive_evidence»
            model.head.fc.weight.zero_()
        if negative_bias is not None:
            model.head.fc.bias.fill_(float(negative_bias))
    model.eval()
    return write_bundle(
        model,
        data_config,
        out,
        image_size=IMAGE_SIZE,
        val_resize="center_crop",
        opset=OPSET,
        calibration=SYNTHETIC_CALIBRATION,
        thresholds=SYNTHETIC_THRESHOLDS,
        metrics=SYNTHETIC_METRICS,
        checkpoint_sha256="0" * 64,
        checkpoint_file=f"none (pesos aleatorios, semilla {seed})",
        git_sha="test",
        extra={
            "kind": "test",
            "backbone": BACKBONE,
            "warning": "PESOS ALEATORIOS: solo para CI y pruebas",
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/test_bundle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--negative-bias", type=float, default=None)
    ap.add_argument("--zero-weights", action="store_true")
    a = ap.parse_args()
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    m = build(out, a.seed, a.negative_bias, a.zero_weights)
    size = sum((out / f).stat().st_size for f in m["files"]) / 1e6
    print(f"→ {out} · {m['model_version']} · {size:.1f} MB · opset {m['onnx']['opset']}")


if __name__ == "__main__":
    main()
