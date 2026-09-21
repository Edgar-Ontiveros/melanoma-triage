# Índice CUDA para `make setup-gpu`. Para torch 2.14 existen cu126 y cu130 (no cu128); ajustar al driver de la instancia.
CUDA ?= cu126
TORCH_CUDA_INDEX = https://download.pytorch.org/whl/$(CUDA)
UV_RUN = uv run --no-sync

# Todos los objetivos son comandos, no archivos: sin .PHONY, un directorio con el mismo nombre
# (p. ej. kaggle/) hace que make diga "is up to date" y no ejecute nada (pasó el 2026-09-17).
# tests/test_makefile.py verifica que cada objetivo esté aquí.
.PHONY: setup setup-gpu lint format test smoke ci docker-api docker-size clean \
	data-verify data-manifest data-dedup data-splits data-resize data-eda data-all \
	kaggle-dataset train baseline-b0 f2-report f3-report f4-report check-notebooks kaggle \
	f5-cam f5-overlap f5-artifacts f5-figures f5-report f5-all

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

## ---- Pipeline de datos F1 (config en configs/data/isic2020.yaml). Orden: verify → manifest →
## dedup → splits → resize → eda. `data-all` los encadena.
data-verify:
	$(UV_RUN) python scripts/verify_download.py
data-manifest:
	$(UV_RUN) python scripts/build_manifest.py
data-dedup:
	$(UV_RUN) python scripts/dedup_report.py
data-splits:
	$(UV_RUN) python scripts/make_splits.py
data-resize:
	$(UV_RUN) python scripts/resize_images.py
data-eda:
	$(UV_RUN) python scripts/eda_report.py
data-all: data-verify data-manifest data-dedup data-splits data-resize data-eda

## Dataset privado de Kaggle (solo manifiesto + splits). Requiere KAGGLE_USERNAME y ~/.kaggle/kaggle.json.
kaggle-dataset:
	scripts/kaggle_dataset.sh $(MODE)

## ---- F2: entrenamiento y líneas base. `make train ARGS="+experiment=b1_resnet50_224 train.seed=0"`.
train:
	$(UV_RUN) python scripts/train.py $(ARGS)
baseline-b0:
	$(UV_RUN) python scripts/baseline_metadata.py
## Reportes de F2 a partir de reports/metrics/b0_val.json y reports/runs/*/metrics.json.
f2-report:
	$(UV_RUN) python scripts/f2_report.py
f3-report:
	$(UV_RUN) python scripts/f3_report.py
f4-report:
	$(UV_RUN) python scripts/f4_report.py
## ---- F5: explicabilidad sobre validación (configs/f5.yaml). Orden: cam → overlap → artifacts →
## figures → report; `f5-all` los encadena (~20 min de CPU).
f5-cam:
	$(UV_RUN) python scripts/f5_cam.py
f5-overlap:
	$(UV_RUN) python scripts/f5_overlap.py
f5-artifacts:
	$(UV_RUN) python scripts/f5_artifacts.py
f5-figures:
	$(UV_RUN) python scripts/f5_figures.py
f5-report:
	$(UV_RUN) python scripts/f5_report.py
f5-all: f5-cam f5-overlap f5-artifacts f5-figures f5-report
## Ejecuta los notebooks de Kaggle en un sandbox local (venv limpio + clon + /kaggle/input sintético).
check-notebooks:
	$(UV_RUN) python scripts/check_notebooks.py
## Corridas en Kaggle por API (configs/kaggle.yaml). Ej.: make kaggle ARGS="push b1-s0 --sha <SHA> --wait"
kaggle:
	$(UV_RUN) python scripts/kaggle_run.py $(ARGS)
