"""API de triage asistido (F6.3, F7.1): FastAPI + onnxruntime sobre el paquete de modelo, y la
interfaz web (build de Vite) servida como archivos estáticos.

Endpoints bajo ``/api``: ``POST /api/predict`` (una imagen), ``GET /api/health``,
``GET /api/model-info``, ``GET /api/limitations`` (docs/f4_limitations.md en Markdown).
La SPA se sirve en ``/`` desde ``WEB_DIST_DIR`` (si existe) con *fallback* a ``index.html``
para las rutas del cliente (``/modelo``). Un solo contenedor, un solo puerto.

Al arrancar se verifica el SHA256 de cada archivo del paquete (``MODEL_BUNDLE_DIR``); si
alguno no coincide la aplicación no levanta. Ningún número del modelo vive en este código.
"""

from __future__ import annotations

import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from services.api.bundle import Bundle, load_bundle
from services.api.inference import MAX_BYTES, InputError, Predictor
from services.api.render import overlay_png_base64

ENV_THREADS = "ORT_INTRA_OP_THREADS"
ENV_WEB_DIST = "WEB_DIST_DIR"
ENV_LIMITATIONS = "LIMITATIONS_MD"
DEFAULT_THREADS = 4
API_PREFIX = "/api"
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEB_DIST = _ROOT / "services" / "web" / "dist"
DEFAULT_LIMITATIONS = _ROOT / "docs" / "f4_limitations.md"

DISCLAIMER = (
    "Herramienta de apoyo al triage de melanoma en imágenes dermatoscópicas, con fines "
    "educativos y de investigación. No es un diagnóstico: indica si una lesión conviene "
    "referirse a valoración especializada en un punto de operación de alta sensibilidad y "
    "baja especificidad. No detecta otros cánceres de piel ni otras enfermedades. No "
    "sustituye la valoración de un dermatólogo. No cuenta con aprobación regulatoria "
    "(COFEPRIS, FDA, CE) y no debe usarse para tomar decisiones clínicas. Las imágenes se "
    "procesan en memoria y no se almacenan."
)


class OperatingPoint(BaseModel):
    sensitivity_target: float
    sensitivity_test: float | None
    specificity_test: float | None


class CamOut(BaseModel):
    status: str = Field(description="'ok' o 'no_positive_evidence'")
    grid: list[list[float]]
    png_base64: str | None
    crop_png_base64: str = Field(description="el recorte de 224 px que vio el modelo, sin mapa")


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


class LimitationsResponse(BaseModel):
    markdown: str
    source: str
    n_sections: int


def _threads() -> int:
    return int(os.environ.get(ENV_THREADS, DEFAULT_THREADS))


def _web_dist(web_dist: Path | None) -> Path | None:
    d = web_dist if web_dist is not None else Path(os.environ.get(ENV_WEB_DIST, DEFAULT_WEB_DIST))
    return d if (d / "index.html").exists() else None


def _limitations_path(path: Path | None) -> Path:
    if path is not None:
        return path
    return Path(os.environ.get(ENV_LIMITATIONS, DEFAULT_LIMITATIONS))


def count_sections(markdown: str) -> int:
    """Bloques no vacíos separados por línea en blanco (encabezados o párrafos)."""
    return len([b for b in re.split(r"\n\s*\n", markdown) if b.strip()])


def build_app(
    bundle_dir: Path | None = None,
    intra_op_threads: int | None = None,
    web_dist: Path | None = None,
    limitations_md: Path | None = None,
) -> FastAPI:
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
        version="F7",
        lifespan=lifespan,
        docs_url=f"{API_PREFIX}/docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
        redoc_url=None,
    )
    api = APIRouter(prefix=API_PREFIX)

    @app.exception_handler(InputError)
    async def _input_error(request: Request, exc: InputError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": exc.detail})

    @api.get("/health", response_model=HealthResponse)
    async def health() -> dict[str, Any]:
        predictor = getattr(app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(status_code=503, detail="el paquete de modelo no cargó")
        return {"status": "ok", "model_version": app.state.bundle.model_version}

    @api.get("/model-info")
    async def model_info() -> dict[str, Any]:
        b: Bundle = app.state.bundle
        return {
            "model_version": b.model_version,
            "manifest": b.manifest,
            "metrics": b.metrics,
            "thresholds": b.thresholds,
            "calibration": dict(b.calibration),
            "preprocess": b.preprocess,
            "disclaimer": DISCLAIMER,
        }

    @api.get("/limitations", response_model=LimitationsResponse)
    async def limitations() -> dict[str, Any]:
        path = _limitations_path(limitations_md)
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail="docs/f4_limitations.md pendiente: las limitaciones las escribe el autor",
            )
        text = path.read_text(encoding="utf-8")
        return {"markdown": text, "source": path.name, "n_sections": count_sections(text)}

    @api.post("/predict", response_model=PredictResponse)
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
        crop_png = overlay_png_base64(pred.crop, None)
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
                "crop_png_base64": crop_png,
            },
            "input": pred.input_info,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "timings_ms": {k: round(v, 2) for k, v in pred.timings_ms.items()},
            "disclaimer": DISCLAIMER,
        }

    app.include_router(api)

    dist = _web_dist(web_dist)
    app.state.web_dist = dist
    if dist is not None:
        dist_resolved = dist.resolve()

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str) -> FileResponse:
            """Archivos del build; cualquier otra ruta (p. ej. /modelo) recibe index.html."""
            if path:
                candidate = (dist_resolved / path).resolve()
                if candidate.is_file() and dist_resolved in candidate.parents:
                    return FileResponse(candidate)
            return FileResponse(dist_resolved / "index.html")

    return app


def create_app() -> FastAPI:
    """Punto de entrada de uvicorn: ``uvicorn services.api.main:create_app --factory``."""
    return build_app()


# ``uvicorn services.api.main:app`` también funciona: el paquete se carga al arrancar (lifespan).
app = build_app()
