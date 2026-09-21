"""Cadena de inferencia de la API: bytes → PIL → preprocesamiento numpy → ONNX → Platt →
decisión en τ95 → CAM. Devuelve además el desglose de tiempos para F6.4."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
from services.api.bundle import Bundle
from services.api.cam import cam_from_features, upsample
from services.api.preprocess import Preprocessor

ACCEPTED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
MAX_BYTES = 10 * 1024 * 1024
MIN_SIDE = 64


class InputError(ValueError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass
class Prediction:
    logit: float
    probability: float
    refer: bool
    cam_grid: np.ndarray
    cam_status: str
    crop: np.ndarray
    input_info: dict[str, Any]
    timings_ms: dict[str, float]


def decode_image(data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    """Valida tipo, tamaño y lado mínimo; devuelve RGB uint8 y la descripción de la entrada."""
    if len(data) > MAX_BYTES:
        raise InputError(413, f"la imagen supera {MAX_BYTES // (1024 * 1024)} MB")
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format or ""
            if fmt not in ACCEPTED_FORMATS:
                raise InputError(
                    415, f"formato {fmt or 'desconocido'} no aceptado; use JPEG, PNG o WebP"
                )
            width, height = img.size
            if min(width, height) < MIN_SIDE:
                raise InputError(422, f"lado mínimo {MIN_SIDE} px; la imagen mide {width}×{height}")
            rgb = np.asarray(img.convert("RGB"))
    except UnidentifiedImageError as e:
        raise InputError(422, "no se pudo abrir el archivo como imagen") from e
    except OSError as e:  # imagen truncada o corrupta
        raise InputError(422, f"imagen ilegible: {e}") from e
    return rgb, {"width": int(width), "height": int(height), "format": fmt}


class Predictor:
    def __init__(self, bundle: Bundle, intra_op_threads: int = 4) -> None:
        self.bundle = bundle
        self.pre = Preprocessor(bundle.preprocess)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(intra_op_threads)
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(bundle.onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = bundle.manifest["onnx"]["input"]
        self.output_names = list(bundle.manifest["onnx"]["outputs"])
        self.platt_a = float(bundle.calibration["a"])
        self.platt_b = float(bundle.calibration["b"])
        self.threshold = float(bundle.operating["threshold"])

    def calibrate(self, logit: float) -> float:
        z = self.platt_a * logit + self.platt_b
        return float(1.0 / (1.0 + np.exp(-z)))

    def predict_array(
        self, rgb: np.ndarray, input_info: dict[str, Any] | None = None
    ) -> Prediction:
        t: dict[str, float] = {}
        t0 = time.perf_counter()
        crop, x = self.pre(rgb)
        t["preprocess"] = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        logit_arr, feats = self.session.run(self.output_names, {self.input_name: x})
        t["onnx"] = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        logit = float(logit_arr[0, 0])
        prob = self.calibrate(logit)
        grid = cam_from_features(feats[0], self.bundle.cam_weights)
        status = "ok" if grid.max() > 0 else "no_positive_evidence"
        t["cam"] = (time.perf_counter() - t2) * 1000
        return Prediction(
            logit=logit,
            probability=prob,
            refer=bool(prob >= self.threshold),
            cam_grid=grid,
            cam_status=status,
            crop=crop,
            input_info=input_info
            or {"width": int(rgb.shape[1]), "height": int(rgb.shape[0]), "format": "array"},
            timings_ms=t,
        )

    def predict_bytes(self, data: bytes) -> Prediction:
        t0 = time.perf_counter()
        rgb, info = decode_image(data)
        decode_ms = (time.perf_counter() - t0) * 1000
        pred = self.predict_array(rgb, info)
        pred.timings_ms = {"decode": decode_ms, **pred.timings_ms}
        return pred

    def cam_upsampled(self, pred: Prediction) -> np.ndarray:
        return upsample(pred.cam_grid, self.pre.size)
