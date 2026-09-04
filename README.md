# melanoma

Clasificador de lesiones cutáneas con redes convolucionales, desarrollado como proyecto de tesis.

## Qué es y qué no es

Esto es una **herramienta educativa y de triage asistido**: un experimento de aprendizaje
automático sobre imágenes dermatoscópicas públicas, con explicabilidad y una API de
demostración.

**No es un dispositivo de diagnóstico.** No está validado clínicamente, no ha sido evaluado
por ninguna autoridad sanitaria y no debe usarse para tomar, confirmar o descartar decisiones
médicas. Cualquier salida del modelo es una probabilidad estadística sobre una imagen, no
una opinión médica. Ante una lesión sospechosa, consulte a un dermatólogo.

## Instalación

Plataforma de desarrollo soportada: **WSL2 con Ubuntu 24.04** (o Ubuntu nativo). Requiere
[uv](https://docs.astral.sh/uv/) y Python 3.12 (uv lo descarga si falta).

### Laptop / CPU (WSL/Ubuntu)

```bash
git clone <url-del-repositorio> melanoma
cd melanoma
make setup        # uv sync --extra train --group dev  (ruedas CPU de PyTorch, del lockfile)
make ci           # lint + pruebas + humo, lo mismo que GitHub Actions
```

`make setup` instala los grupos `core`, `train` y `dev` con las ruedas **CPU** de PyTorch
fijadas en `uv.lock` (índice `https://download.pytorch.org/whl/cpu`).

### Instancia con GPU (entrenamiento)

```bash
make setup-gpu CUDA=cu126
```

Ejecuta `make setup` y después reinstala `torch` y `torchvision` desde
`https://download.pytorch.org/whl/cu126` (o el índice que indique `CUDA=`). El lockfile no
cambia: fijar CUDA en él rompería la instalación local. Verificar con:

```bash
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Prueba de humo

```bash
make smoke        # o: uv run python scripts/smoke_train.py
```

Entrena `resnet18` sin pesos preentrenados sobre 50 imágenes de ruido generadas en memoria,
2 épocas a 224 px, batch 8, en CPU. No descarga nada. Debe terminar en menos de 2 minutos.
Lee `configs/smoke.yaml` con Hydra, exactamente como lo haría una corrida real, así que un
error de configuración aparece aquí y no a los 40 minutos de una GPU rentada.

## Estructura del repositorio

```
.github/workflows/   CI: ruff, pytest, humo, build de la API y límite de 400 MB
configs/             Configuración Hydra: data/, model/, train/, composiciones raíz (config.yaml, prepare.yaml)
src/melanoma/        Paquete Python: data (manifiesto, dedup, splits, resize), models (fábrica timm), train, eval, explain, export, utils
services/api/        API de inferencia FastAPI + onnxruntime (grupo `api`, sin torch ni core) y su Dockerfile
apps/web/            Frontend (fase posterior)
data/manifests/      isic2020.csv, una fila por imagen con SHA256 (versionado)
data/splits/         train.txt, val.txt, test.txt y SHA256SUMS (versionados)
data/raw/, data/processed/   Imágenes originales y redimensionadas (NO versionadas)
reports/             dedup_report.md, eda.md, splits_report.md, manifest_report.md, resize_report.md y figuras
notebooks/           kaggle_setup.ipynb: plantilla que verifica la procedencia en Kaggle
kaggle/              Metadatos del dataset privado de Kaggle (manifiesto + splits)
scripts/             Pipeline de datos, smoke_train.py y utilidades de CI
tests/               Pruebas rápidas sin descargas (las que necesitan datos se saltan si faltan)
docs/                DATA.md (licencia, procedencia), hardware.md, specs/ por fase
logs/                test_set_access.log (versionado) y salidas de corridas (no versionadas)
```

Regla del proyecto: **cero constantes de configuración en `src/`**. Tamaños de imagen,
tasas de aprendizaje y rutas viven solo en `configs/`. La normalización de píxeles
(`mean`, `std`, `input_size`) proviene únicamente del `data_config` que devuelve
`melanoma.models.factory.build_model`.

## Datos: ISIC 2020

Detalles de licencia, atribución y procedencia en `docs/DATA.md`. El pipeline completo de F1
corre con `make data-all` (o paso por paso: `data-verify`, `data-manifest`, `data-dedup`,
`data-splits`, `data-resize`, `data-eda`) y está parametrizado en `configs/data/isic2020.yaml`.

Descarga (fuente oficial, ~24.7 GB; el ZIP se verifica por tamaño y las imágenes por SHA256):

```bash
mkdir -p data/raw/isic2020 && cd data/raw/isic2020
for f in ISIC_2020_Training_JPEG.zip ISIC_2020_Training_GroundTruth_v2.csv ISIC_2020_Training_Duplicates.csv; do
  curl -L -O "https://isic-challenge-data.s3.amazonaws.com/2020/$f"
done
```

### Splits agrupados por paciente

70 / 15 / 15 por paciente, estratificados por «paciente con al menos un melanoma», semilla
`20260904`. Un `image_id` por línea, ordenados. Conteos alcanzados en `reports/splits_report.md`.
SHA256 de los archivos versionados (verificado por `tests/test_splits.py`):

```
701333b2450c363b1ab1b5a9422d3d223d243e1b3af2d643b05f6b86f711968c  train.txt
b7f727d6cf70d7a12be136e3b7046b42c725eaf5541aadc0723fbac09941ffb1  val.txt
7dedecdd0ebf1e8997eb4be12fe71af04475de2bd1b0da5d6fde8440a7742316  test.txt
```

### Conjunto de prueba bloqueado

`data/splits/test.txt` se usa **como máximo dos veces en todo el proyecto**, ambas desde
`scripts/evaluate_test.py` (F4), y cada acceso se anota en `logs/test_set_access.log`. Ningún
módulo de `src/` puede referenciarlo: lo vigila `tests/test_test_split_isolation.py` en CI.

### Kaggle

El entrenamiento corre en Kaggle. Ahí se adjunta la distribución oficial de la competencia
(`siim-isic-melanoma-classification`) y un dataset privado con solo el manifiesto y los splits
(`make kaggle-dataset KAGGLE_USERNAME=<usuario>`). `notebooks/kaggle_setup.ipynb` verifica el
SHA256 de 200 imágenes contra el manifiesto y falla si alguno no coincide.

## Licencias y atribución

- **Código:** MIT (ver `LICENSE`).
- **Datos:** el ISIC 2020 Challenge Dataset se distribuye bajo
  [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Cita requerida:

  > International Skin Imaging Collaboration. SIIM-ISIC 2020 Challenge Dataset.
  > International Skin Imaging Collaboration. https://doi.org/10.34970/2020-ds01 (2020).
  >
  > Rotemberg, V., Kurtansky, N., Betz-Stablein, B., et al. A patient-centric dataset of
  > images and metadata for identifying melanomas using clinical context.
  > *Sci Data* 8, 34 (2021). https://doi.org/10.1038/s41597-021-00815-z

- **Pesos y artefactos derivados:** cualquier modelo entrenado sobre datos CC-BY-NC hereda
  la restricción **no comercial**. Los checkpoints, archivos ONNX y el despliegue de la API
  tienen que ser no comerciales y mantener la atribución anterior.

## Estado del proyecto

| Fase | Contenido | Estado |
|---|---|---|
| F0 | Esqueleto, entorno, CI, prueba de humo | Cerrada (`docs/specs/F0.md`) |
| F1 | Manifiesto, deduplicación, splits por paciente, bloqueo del test, Kaggle | Cerrada en local; pendiente correr `notebooks/kaggle_setup.ipynb` en Kaggle (`docs/specs/F1.md`) |
| F2 | Entrenamiento y evaluación | Pendiente |
| F3 | GPU rentada, exportación ONNX, API | Pendiente |
| F4 | Grad-CAM, evaluación externa (DDI) | Pendiente |

Diagnóstico del hardware de desarrollo: `docs/hardware.md`.
