# Imagen de la API de inferencia. Solo core + api (fastapi, uvicorn, onnxruntime).
# Nunca torch/timm/lightning: el CI falla si la imagen supera 400 MB.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /build
COPY pyproject.toml uv.lock ./
# Exporta solo las dependencias de core + api (sin el proyecto, sin dev, sin train).
RUN uv export --frozen --no-dev --no-emit-project --extra api -o requirements.txt \
    && uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache -r requirements.txt

FROM python:3.12-slim-bookworm
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY services/api /app/services/api
# F0: solo verifica que el paquete importa. Los endpoints llegan en F3.
CMD ["python", "-c", "import services.api; print('melanoma-api OK')"]
