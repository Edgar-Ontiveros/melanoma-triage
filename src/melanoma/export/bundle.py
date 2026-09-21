"""Exportación del paquete de modelo (F6.1): ONNX con dos salidas, preprocesamiento de timm,
calibración y umbrales de F4, pesos de la capa lineal para el CAM, métricas del test y un
manifiesto con SHA256 de todo.

La API (services/api) lee únicamente este paquete: ningún número del modelo vive en su código.
Aquí tampoco se escribe ninguno a mano: `preprocess.json` sale de `resolve_data_config`,
`calibration.json`/`thresholds.json` de reports/f4_calibration.json, `metrics.json` de
reports/f4_test_results.json y `cam_weights.npy` del checkpoint.

Exportador: el de TorchScript (`dynamo=False`). El exportador nuevo (torch.export) no puede
producir opset 17: genera opset 18 y el conversor de versiones falla en `Pad`. La spec fija
opset 17, así que se usa el exportador clásico; la paridad se verifica en F6.2.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from melanoma.models.factory import BackboneClassifier

BUNDLE_FILES = (
    "model.onnx",
    "preprocess.json",
    "calibration.json",
    "thresholds.json",
    "cam_weights.npy",
    "metrics.json",
)
INPUT_NAME = "image"
OUTPUT_NAMES = ("logit", "features")


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ExportModel(nn.Module):
    """``x → (logit [N, 1], features [N, C, H, W])``: la salida de ``backbone.bn2`` (lo que
    entra al pooling) y el logit calculado con la misma cabeza. Es el contrato de F5."""

    def __init__(self, clf: BackboneClassifier) -> None:
        super().__init__()
        self.clf = clf

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.clf.backbone.forward_features(x)
        return self.clf.head(feats.mean(dim=(2, 3))), feats


def model_version(git_sha: str, checkpoint_sha256: str) -> str:
    return f"{git_sha[:7]}-{checkpoint_sha256[:8]}"


def stage1_spec(long_side: int, jpeg_quality: int) -> dict:
    """Etapa 1 = el redimensionado de F1 (``melanoma.data.resize.resize_one``): lado largo a
    ``long_side`` con Lanczos de PIL, sin agrandar, JPEG a ``jpeg_quality``, con ``draft``.
    Los números vienen del bloque ``data.resize`` de ``configs/data/isic2020.yaml``."""
    return {
        "long_side": int(long_side),
        "filter": "lanczos",
        "jpeg_quality": int(jpeg_quality),
        "draft": True,
        "noop_if_long_side_leq": int(long_side),
        "source": "configs/data/isic2020.yaml (data.resize) + melanoma.data.resize.resize_one",
    }


def preprocess_spec(
    data_config: dict[str, Any], image_size: int, val_resize: str, stage1: dict
) -> dict:
    """Todo sale del ``data_config`` de timm salvo el lado (config de F3/F4), la geometría y
    la etapa 1 (config de F1)."""
    return {
        "stage1": dict(stage1),
        "input_size": int(image_size),
        "interpolation": str(data_config["interpolation"]),
        "crop_pct": float(data_config.get("crop_pct", 1.0)),
        "mean": [float(m) for m in data_config["mean"]],
        "std": [float(s) for s in data_config["std"]],
        "val_resize": str(val_resize),
        "pixel_max": 255.0,
        "channel_order": "RGB",
        "layout": "NCHW",
        "backbone_input_size": [int(v) for v in data_config["input_size"]],
    }


def calibration_spec(f4: dict) -> dict:
    return {
        "method": "platt",
        "a": float(f4["platt"]["a"]),
        "b": float(f4["platt"]["b"]),
        "fitted_on": "validation",
        "n_images": int(f4["n_images"]),
        "n_positives": int(f4["n_positives"]),
        "date": f4["date"],
        "source": "reports/f4_calibration.json",
    }


def thresholds_spec(f4: dict) -> dict:
    out: dict[str, Any] = {"operating": "tau_95", "source": "reports/f4_calibration.json"}
    for key in ("tau_95", "tau_90"):
        t = f4["thresholds"][key]
        o = t["on_full_set"]
        out[key] = {
            "tau": float(t["tau"]),
            "sensitivity_target": float(t["sensitivity_target"]),
            "p05": float(t["p05"]),
            "p95": float(t["p95"]),
            "validation": {
                "sensitivity": float(o["sensitivity"]),
                "specificity": float(o["specificity"]),
                "ppv": float(o["ppv"]),
                "npv": float(o["npv"]),
            },
        }
    return out


def metrics_spec(f4_test: dict) -> dict:
    m, ci = f4_test["metrics"], f4_test["ci"]
    c95 = f4_test["confusion_tau_95"]
    return {
        "split": "test",
        "accessed_once": True,
        "date": f4_test["date"],
        "n_images": int(f4_test["n_images"]),
        "n_positives": int(f4_test["n_positives"]),
        "n_patients": int(f4_test["n_patients"]),
        "auroc": {k: float(ci["auroc"][k]) for k in ("point", "lo", "hi")},
        "auprc": {k: float(ci["auprc"][k]) for k in ("point", "lo", "hi")},
        "prevalence": float(m["prevalence"]),
        "tau_95": {
            "threshold": float(c95["threshold"]),
            "sensitivity": float(c95["sensitivity"]),
            "specificity": float(c95["specificity"]),
            "ppv": float(c95["ppv"]),
            "npv": float(c95["npv"]),
            "tp": int(c95["tp"]),
            "fn": int(c95["fn"]),
            "fp": int(c95["fp"]),
            "tn": int(c95["tn"]),
        },
        "ci": {"level": 0.95, "unit": "patient", "n_resamples": int(ci["auroc"]["n_resamples"])},
        "source": "reports/f4_test_results.json",
    }


def export_onnx(clf: BackboneClassifier, path: Path, image_size: int, opset: int) -> dict:
    """Exporta en ``eval()``, verifica con ``onnx.checker`` y con una pasada en onnxruntime.

    Devuelve un resumen (opset real, tamaño, diferencia máxima contra torch en un lote
    aleatorio de 2 imágenes y comprobación de lote dinámico con 3).
    """
    clf.eval()
    wrapped = ExportModel(clf).eval()
    x = torch.randn(2, 3, image_size, image_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            wrapped,
            (x,),
            str(path),
            input_names=[INPUT_NAME],
            output_names=list(OUTPUT_NAMES),
            opset_version=opset,
            dynamo=False,
            dynamic_axes={
                INPUT_NAME: {0: "batch"},
                OUTPUT_NAMES[0]: {0: "batch"},
                OUTPUT_NAMES[1]: {0: "batch"},
            },
            do_constant_folding=True,
        )
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    opsets = {o.domain or "ai.onnx": int(o.version) for o in model.opset_import}
    if opsets.get("ai.onnx") != opset:
        raise RuntimeError(f"opset exportado {opsets} ≠ {opset}")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref_logit, ref_feat = wrapped(x)
    out_logit, out_feat = sess.run(list(OUTPUT_NAMES), {INPUT_NAME: x.numpy()})
    diff = max(
        float(np.abs(out_logit - ref_logit.numpy()).max()),
        float(np.abs(out_feat - ref_feat.numpy()).max()),
    )
    x3 = torch.randn(3, 3, image_size, image_size)
    o3 = sess.run([OUTPUT_NAMES[0]], {INPUT_NAME: x3.numpy()})[0]
    if o3.shape != (3, 1) or tuple(out_feat.shape[1:]) != tuple(ref_feat.shape[1:]):
        raise RuntimeError("el lote dinámico o la forma de features no es la esperada")
    return {
        "opset": opsets,
        "bytes": path.stat().st_size,
        "max_abs_diff_random_batch": diff,
        "features_shape": [int(s) for s in ref_feat.shape[1:]],
        "input_size": int(image_size),
        "exporter": "torchscript (dynamo=False)",
    }


def write_bundle(
    clf: BackboneClassifier,
    data_config: dict[str, Any],
    out_dir: Path,
    *,
    image_size: int,
    val_resize: str,
    stage1: dict,
    opset: int,
    calibration: dict,
    thresholds: dict,
    metrics: dict,
    checkpoint_sha256: str,
    checkpoint_file: str,
    git_sha: str,
    extra: dict | None = None,
) -> dict:
    """Escribe los siete archivos en ``out_dir`` y devuelve el manifiesto."""
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_info = export_onnx(clf, out_dir / "model.onnx", image_size, opset)
    w = clf.head.fc.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
    np.save(out_dir / "cam_weights.npy", w)
    files = {
        "preprocess.json": preprocess_spec(data_config, image_size, val_resize, stage1),
        "calibration.json": calibration,
        "thresholds.json": thresholds,
        "metrics.json": metrics,
    }
    for name, payload in files.items():
        (out_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest = {
        "model_version": model_version(git_sha, checkpoint_sha256),
        "git_sha": git_sha,
        "checkpoint": {"file": checkpoint_file, "sha256": checkpoint_sha256},
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "onnx": {
            "input": INPUT_NAME,
            "outputs": list(OUTPUT_NAMES),
            "opset": onnx_info["opset"],
            "exporter": onnx_info["exporter"],
            "features_shape": onnx_info["features_shape"],
            "max_abs_diff_random_batch": onnx_info["max_abs_diff_random_batch"],
        },
        "cam": {
            "formula": "ReLU(sum_k w_k * A_k) / max over features; zero map = no positive evidence",
            "layer": "backbone.bn2",
            "n_weights": int(w.size),
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "files": {name: sha256_file(out_dir / name) for name in BUNDLE_FILES},
    }
    if extra:
        manifest.update(extra)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
