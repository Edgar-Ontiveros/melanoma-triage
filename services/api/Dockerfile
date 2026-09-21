# Imagen de la API de inferencia (F6.5). Solo el grupo `api` del lockfile (fastapi, uvicorn,
# pydantic, onnxruntime, pillow, python-multipart, numpy). Sin `core`: nada de pandas, hydra
# ni omegaconf. Nunca torch/timm/lightning: el CI falla si la imagen supera 400 MB.
#
# El paquete de modelo NO va en la imagen: se monta en /bundle (MODEL_BUNDLE_DIR). En CI se
# monta el paquete de prueba (make test-bundle); en despliegue (F8), el real del release.
#   docker run --rm -p 8000:8000 -v $PWD/models/bundle:/bundle:ro melanoma-api
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /build
COPY pyproject.toml uv.lock ./
# Exporta exactamente el grupo `api` (sin el proyecto ni sus dependencias, sin dev, sin train).
RUN uv export --frozen --only-group api --no-emit-project -o requirements.txt \
    && uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache -r requirements.txt

FROM python:3.12-slim-bookworm
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_BUNDLE_DIR=/bundle \
    ORT_INTRA_OP_THREADS=4
RUN groupadd --system api && useradd --system --gid api --no-create-home api \
    && mkdir -p /app /bundle && chown -R api:api /app /bundle
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=api:api services/api /app/services/api
USER api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
