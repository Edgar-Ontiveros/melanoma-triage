"""API de triage asistido (F6.3): FastAPI + onnxruntime sobre el paquete de modelo.

Endpoints: ``POST /predict`` (una imagen), ``GET /health``, ``GET /model-info``.
Al arrancar se verifica el SHA256 de cada archivo del paquete (``MODEL_BUNDLE_DIR``); si
alguno no coincide la aplicación no levanta. Ningún número del modelo vive en este código.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from services.api.bundle import Bundle, load_bundle
from services.api.inference import MAX_BYTES, InputError, Predictor
from services.api.render import overlay_png_base64

ENV_THREADS = "ORT_INTRA_OP_THREADS"
DEFAULT_THREADS = 4

DISCLAIMER = (
    "Herramienta de apoyo al triage de melanoma en imágenes dermatoscópicas, con fines "
    "educativos y de investigación. No es un diagnóstico: indica si una lesión conviene "
    "referirse a valoración especializada en un punto de operación de alta sensibilidad y "
    "baja especificidad. No detecta otros cánceres de piel ni otras enfermedades. No "
    "sustituye la valoración de un dermatólogo. No cuenta con aprobación regulatoria "
    "(COFEPRIS, FDA, CE) y no debe usarse para tomar decisiones clínicas."
)


class OperatingPoint(BaseModel):
    sensitivity_target: float
    sensitivity_test: float | None
    specificity_test: float | None


class CamOut(BaseModel):
    status: str = Field(description="'ok' o 'no_positive_evidence'")
    grid: list[list[float]]
    png_base64: str | None


class InputInfo(BaseModel):
    width: int
    height: int
    format: str


class PredictResponse(BaseModel):
    model_version: str
    probability: float = Field(ge=0.0, le=1.0, description="probabilidad calibrada (Platt)")
    refer: bool
    threshold: float
    operating_point: OperatingPoint
    cam: CamOut
    input: InputInfo
    latency_ms: float
    timings_ms: dict[str, float]
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_version: str


def _threads() -> int:
    return int(os.environ.get(ENV_THREADS, DEFAULT_THREADS))


def build_app(bundle_dir: Path | None = None, intra_op_threads: int | None = None) -> FastAPI:
    """Crea la aplicación cargando el paquete de ``bundle_dir`` (o ``MODEL_BUNDLE_DIR``)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bundle: Bundle = load_bundle(bundle_dir)  # BundleError → la app no arranca
        app.state.bundle = bundle
        app.state.predictor = Predictor(bundle, intra_op_threads or _threads())
        yield

    app = FastAPI(
        title="melanoma-triage API",
        description="Triage asistido de melanoma (educativo, no diagnóstico).",
        version="F6",
        lifespan=lifespan,
    )

    @app.exception_handler(InputError)
    async def _input_error(request: Request, exc: InputError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": exc.detail})

    @app.get("/health", response_model=HealthResponse)
    async def health() -> dict[str, Any]:
        predictor = getattr(app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(status_code=503, detail="el paquete de modelo no cargó")
        return {"status": "ok", "model_version": app.state.bundle.model_version}

    @app.get("/model-info")
    async def model_info() -> dict[str, Any]:
        b: Bundle = app.state.bundle
        return {
            "model_version": b.model_version,
            "manifest": b.manifest,
            "metrics": b.metrics,
            "thresholds": b.thresholds,
            "calibration": {k: v for k, v in b.calibration.items() if k != "a" and k != "b"}
            | {"a": b.calibration["a"], "b": b.calibration["b"]},
            "preprocess": b.preprocess,
            "disclaimer": DISCLAIMER,
        }

    @app.post("/predict", response_model=PredictResponse)
    async def predict(file: Annotated[UploadFile, File(...)]) -> dict[str, Any]:
        t0 = time.perf_counter()
        data = await file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise InputError(413, f"la imagen supera {MAX_BYTES // (1024 * 1024)} MB")
        if not data:
            raise InputError(422, "archivo vacío")
        predictor: Predictor = app.state.predictor
        pred = predictor.predict_bytes(data)
        t1 = time.perf_counter()
        png = None
        if pred.cam_status == "ok":
            png = overlay_png_base64(pred.crop, predictor.cam_upsampled(pred))
        pred.timings_ms["render"] = (time.perf_counter() - t1) * 1000
        b: Bundle = app.state.bundle
        op = b.operating
        return {
            "model_version": b.model_version,
            "probability": pred.probability,
            "refer": pred.refer,
            "threshold": op["threshold"],
            "operating_point": {
                "sensitivity_target": op["sensitivity_target"],
                "sensitivity_test": op["sensitivity_test"],
                "specificity_test": op["specificity_test"],
            },
            "cam": {
                "status": pred.cam_status,
                "grid": [[round(float(v), 4) for v in row] for row in pred.cam_grid],
                "png_base64": png,
            },
            "input": pred.input_info,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "timings_ms": {k: round(v, 2) for k, v in pred.timings_ms.items()},
            "disclaimer": DISCLAIMER,
        }

    return app


def create_app() -> FastAPI:
    """Punto de entrada de uvicorn: ``uvicorn services.api.main:create_app --factory``."""
    return build_app()


# ``uvicorn services.api.main:app`` también funciona: el paquete se carga al arrancar (lifespan).
app = build_app()
