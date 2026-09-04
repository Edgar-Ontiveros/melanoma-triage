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
configs/             Configuración Hydra: data/, model/, train/ y composiciones raíz
src/melanoma/        Paquete Python: data, models (fábrica timm), train, eval, explain, export, utils
services/api/        API de inferencia FastAPI + onnxruntime (sin torch) y su Dockerfile
apps/web/            Frontend (fase posterior)
data/manifests/      Manifiestos CSV del dataset (no versionados)
data/splits/         Particiones train/val/test (no versionadas)
scripts/             smoke_train.py y utilidades de CI
tests/               Pruebas rápidas sin descargas
docs/                hardware.md, specs/ por fase
logs/                Salidas de corridas (no versionadas)
```

Regla del proyecto: **cero constantes de configuración en `src/`**. Tamaños de imagen,
tasas de aprendizaje y rutas viven solo en `configs/`. La normalización de píxeles
(`mean`, `std`, `input_size`) proviene únicamente del `data_config` que devuelve
`melanoma.models.factory.build_model`.

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
| F0 | Esqueleto, entorno, CI, prueba de humo | En curso (ver `docs/specs/F0.md`) |
| F1 | Manifiestos, splits por paciente, DataModule | Pendiente |
| F2 | Entrenamiento y evaluación | Pendiente |
| F3 | GPU rentada, exportación ONNX, API | Pendiente |
| F4 | Grad-CAM, evaluación externa (DDI) | Pendiente |

Diagnóstico del hardware de desarrollo: `docs/hardware.md`.
