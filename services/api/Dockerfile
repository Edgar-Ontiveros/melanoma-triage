# Imagen de la API + interfaz (F6.5, F7.1). Tres etapas: build de la web (node), dependencias
# del grupo `api` (uv) y la imagen final (python slim). Sin `core`: nada de pandas, hydra ni
# omegaconf. Nunca torch/timm/lightning: el CI falla si la imagen supera 400 MB.
#
# El paquete de modelo NO va en la imagen: se monta en /bundle (MODEL_BUNDLE_DIR). En CI se
# monta el paquete de prueba (make test-bundle); en despliegue (F8), el real del release.
#   docker run --rm -p 8000:8000 -v $PWD/models/bundle:/bundle:ro melanoma-api
FROM node:24-bookworm-slim AS web
WORKDIR /web
COPY services/web/package.json services/web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY services/web/ ./
RUN npm run build

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
    WEB_DIST_DIR=/app/web \
    LIMITATIONS_MD=/app/docs/f4_limitations.md \
    ORT_INTRA_OP_THREADS=4
RUN groupadd --system api && useradd --system --gid api --no-create-home api \
    && mkdir -p /app /bundle && chown -R api:api /app /bundle
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=api:api services/api /app/services/api
COPY --chown=api:api --from=web /web/dist /app/web
# docs/ trae f4_limitations.md (la ficha del modelo lo sirve); si falta, /api/limitations da 404.
COPY --chown=api:api docs /app/docs
USER api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
