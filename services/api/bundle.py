"""Carga del paquete de modelo (F6.3). La API no conoce ningún número del modelo: todo se
lee de aquí, y antes de servir se verifica el SHA256 de cada archivo contra el manifiesto.
Un paquete a medias o alterado no levanta."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ENV_BUNDLE_DIR = "MODEL_BUNDLE_DIR"
REQUIRED_FILES = (
    "model.onnx",
    "preprocess.json",
    "calibration.json",
    "thresholds.json",
    "cam_weights.npy",
    "metrics.json",
)


class BundleError(RuntimeError):
    """El paquete no existe, está incompleto o su SHA256 no coincide."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(bundle_dir: Path) -> dict[str, Any]:
    """Devuelve el manifiesto si todos los archivos existen y sus SHA256 coinciden."""
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise BundleError(f"falta {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    missing = [f for f in REQUIRED_FILES if f not in files]
    if missing:
        raise BundleError(f"el manifiesto no lista {missing}")
    for name, expected in files.items():
        path = bundle_dir / name
        if not path.exists():
            raise BundleError(f"falta {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise BundleError(f"SHA256 de {name} no coincide: {actual[:12]}… ≠ {expected[:12]}…")
    return manifest


@dataclass(frozen=True)
class Bundle:
    dir: Path
    manifest: dict[str, Any]
    preprocess: dict[str, Any]
    calibration: dict[str, Any]
    thresholds: dict[str, Any]
    metrics: dict[str, Any]
    cam_weights: np.ndarray

    @property
    def model_version(self) -> str:
        return str(self.manifest["model_version"])

    @property
    def onnx_path(self) -> Path:
        return self.dir / "model.onnx"

    @property
    def operating(self) -> dict[str, Any]:
        """Umbral de operación (τ95) con su sensibilidad objetivo y las métricas del test."""
        key = self.thresholds["operating"]
        t = self.thresholds[key]
        test = self.metrics.get(key, {})
        return {
            "key": key,
            "threshold": float(t["tau"]),
            "sensitivity_target": float(t["sensitivity_target"]),
            "sensitivity_test": test.get("sensitivity"),
            "specificity_test": test.get("specificity"),
        }


def bundle_dir_from_env() -> Path:
    value = os.environ.get(ENV_BUNDLE_DIR)
    if not value:
        raise BundleError(f"define {ENV_BUNDLE_DIR} con la ruta del paquete de modelo")
    return Path(value)


def load_bundle(bundle_dir: Path | None = None) -> Bundle:
    d = bundle_dir if bundle_dir is not None else bundle_dir_from_env()
    if not d.is_dir():
        raise BundleError(f"{d} no es un directorio")
    manifest = verify_manifest(d)

    def read(name: str) -> dict[str, Any]:
        return json.loads((d / name).read_text(encoding="utf-8"))

    w = np.load(d / "cam_weights.npy").astype(np.float32).reshape(-1)
    return Bundle(
        dir=d,
        manifest=manifest,
        preprocess=read("preprocess.json"),
        calibration=read("calibration.json"),
        thresholds=read("thresholds.json"),
        metrics=read("metrics.json"),
        cam_weights=w,
    )
