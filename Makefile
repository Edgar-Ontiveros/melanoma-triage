# Índice CUDA para `make setup-gpu`. Para torch 2.14 existen cu126 y cu130 (no cu128); ajustar al driver de la instancia.
CUDA ?= cu126
TORCH_CUDA_INDEX = https://download.pytorch.org/whl/$(CUDA)
UV_RUN = uv run --no-sync

.PHONY: setup setup-gpu lint format test smoke ci docker-api docker-size clean

## Entorno local (WSL/Ubuntu): core + train + dev con ruedas CPU del lockfile.
setup:
	uv sync --extra train --group dev

## Instancia de entrenamiento: igual, pero reinstala torch/torchvision con ruedas CUDA.
## No toca uv.lock: las ruedas CUDA se instalan por encima del entorno sincronizado.
setup-gpu: setup
	uv pip install --reinstall --index-url $(TORCH_CUDA_INDEX) torch torchvision

lint:
	$(UV_RUN) ruff format --check .
	$(UV_RUN) ruff check .

format:
	$(UV_RUN) ruff format .
	$(UV_RUN) ruff check --fix .

test:
	$(UV_RUN) pytest --cov --cov-report=term-missing

smoke:
	$(UV_RUN) python scripts/smoke_train.py

## Lo mismo que corre GitHub Actions (menos el build de Docker, ver docker-api).
ci: lint test smoke

## Imagen de la API. Requiere Docker con integración WSL activa.
docker-api:
	docker build -f services/api/Dockerfile -t melanoma-api .

## Falla si la imagen supera 400 MB.
docker-size: docker-api
	@python3 scripts/check_image_size.py melanoma-api 400

clean:
	rm -rf outputs .pytest_cache .ruff_cache .coverage htmlcov
