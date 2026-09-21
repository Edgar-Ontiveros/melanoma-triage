# API de triage asistido (F6)

Servicio FastAPI + onnxruntime que sirve el modelo final (A1, EfficientNetV2-S a 224 px)
**sin PyTorch**, con exactamente los números validados en F4 y el mapa de calor de F5
calculado en numpy. Código en `services/api/`; dependencias = grupo `api` de `pyproject.toml`
(fastapi, uvicorn, onnxruntime, numpy, pillow, pydantic, python-multipart y
opencv-python-headless para el redimensionado). Nada más.

**Principio:** la API no conoce ningún número del modelo. Todo lo lee del paquete de modelo
(`MODEL_BUNDLE_DIR`), verifica el SHA256 de cada archivo contra `manifest.json` al arrancar
y se niega a levantar si alguno no coincide (`tests/test_api.py::test_app_refuses_to_start_with_tampered_bundle`).

## El paquete de modelo

`models/bundle/` (fuera de git; release asset con SHA256), generado por
`make export-bundle` (`scripts/export_bundle.py`, `melanoma.export.bundle`):

| Archivo | Contenido | Origen |
|---|---|---|
| `model.onnx` | Backbone + cabeza, opset 17, lote dinámico, dos salidas: `logit [N, 1]` y `features [N, 1280, 7, 7]` (salida de `backbone.bn2`) | checkpoint de F3 en `eval()`, exportador TorchScript |
| `preprocess.json` | `stage1` (lado largo 512, Lanczos, JPEG 95: el redimensionado de F1), `input_size` 224, `interpolation` bicubic, `mean`/`std` (0.5, 0.5, 0.5), `val_resize` center_crop | `configs/data/isic2020.yaml` (resize) y `timm.data.resolve_data_config` |
| `calibration.json` | Platt `a`, `b` ajustados sobre validación | `reports/f4_calibration.json` |
| `thresholds.json` | τ95 (operación) y τ90 con sensibilidad/especificidad en validación | `reports/f4_calibration.json` |
| `cam_weights.npy` | los 1280 pesos de la capa lineal | checkpoint |
| `metrics.json` | AUC-ROC, AUPRC (IC 95 % por paciente), sens/spec/VPP/VPN en τ95 **del test** | `reports/f4_test_results.json` |
| `manifest.json` | `model_version`, SHA256 de cada archivo, SHA del checkpoint, git SHA, fecha, opset, versiones | generado |

`model_version` = `{git_sha_corto}-{checkpoint_sha_8}`; el paquete actual es
`4e7ab61-e0ecf981`. Paridad con PyTorch verificada en `reports/f6_parity.json`
(`make verify-bundle`).

Para CI y Docker existe un **paquete de prueba** con pesos aleatorios y la misma estructura
(`make test-bundle` → `models/test_bundle/`, `manifest.kind = "test"`, métricas marcadas
como sintéticas). Nunca se sirve.

## Levantar la API

```bash
make setup                                   # instala también el grupo api
make export-bundle                           # necesita data/models/<checkpoint> (o bajar el release)
make api                                     # uvicorn en http://127.0.0.1:8000  (MODEL_BUNDLE_DIR=models/bundle)

# Docker (imagen de 152 MB; el paquete se monta, no va dentro)
make docker-api
docker run --rm -p 8000:8000 -v "$PWD/models/bundle:/bundle:ro" melanoma-api
```

Variables: `MODEL_BUNDLE_DIR` (obligatoria), `ORT_INTRA_OP_THREADS` (4 por defecto).
Documentación interactiva en `/docs` (OpenAPI).

## Endpoints

### `POST /predict`

Multipart con un único campo `file` (JPEG, PNG o WebP; ≤ 10 MB; lado mínimo 64 px).

```bash
curl -s -F "file=@data/processed/isic2020_512/ISIC_0096201.jpg" http://127.0.0.1:8000/predict
```

Respuesta (ejemplo real con el paquete `4e7ab61-e0ecf981`, imagen de 512 px; `png_base64` y `grid` recortados; con el original de 1872 × 1053 px de la misma lesión la probabilidad es idéntica, 0.015491):

```json
{
  "model_version": "4e7ab61-e0ecf981",
  "probability": 0.015490515414014127,
  "refer": true,
  "threshold": 0.0038890622237550752,
  "operating_point": {
    "sensitivity_target": 0.95,
    "sensitivity_test": 0.9651162790697675,
    "specificity_test": 0.4283778552071235
  },
  "cam": {
    "status": "ok",
    "grid": [[0.21, 0.70, 0.92, 0.63, 0.37, 0.08, 0.0], "… 7 filas × 7 columnas, valores en [0, 1], máximo 1"],
    "png_base64": "iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACV…"
  },
  "input": {"width": 512, "height": 288, "format": "JPEG"},
  "latency_ms": 46.1,
  "timings_ms": {"decode_stage1": 5.37, "preprocess": 3.27, "onnx": 28.81, "cam": 0.14, "render": 7.64},
  "disclaimer": "Herramienta de apoyo al triage de melanoma en imágenes dermatoscópicas, con fines educativos y de investigación. No es un diagnóstico: …"
}
```

| Campo | Significado |
|---|---|
| `probability` | probabilidad **calibrada** (Platt de F4). La cruda no se devuelve nunca. |
| `refer` | `probability >= threshold` con τ95. Booleano de triage («conviene referir»), no un diagnóstico. |
| `threshold` | τ95 del paquete (0.0039): sensibilidad objetivo 0.95 sobre validación. |
| `operating_point` | sensibilidad objetivo y sensibilidad/especificidad **medidas en el test** (F4, un solo acceso). |
| `cam.status` | `"ok"`, o `"no_positive_evidence"` cuando ninguna celda del mapa es positiva (181 de 4,963 imágenes de validación en F5). En ese caso `grid` es todo ceros y `png_base64` es `null`: un mapa nulo nunca se normaliza para «que se vea algo». |
| `cam.grid` | CAM 7 × 7 = `ReLU(Σ_k w_k · A_k) / máx`, calculado en numpy a partir de `features` y `cam_weights.npy` (equivalente a Grad-CAM, verificado en F5). |
| `cam.png_base64` | el mapa interpolado a 224 × 224 y superpuesto sobre **el recorte que vio el modelo** (lado corto a 224 + recorte central), no sobre la imagen original. PNG en base64. |
| `input` | dimensiones y formato del archivo recibido. |
| `latency_ms`, `timings_ms` | tiempo dentro del endpoint y desglose: `decode_stage1` (decodificar + etapa 1), `preprocess` (etapa 2), `onnx`, `cam`, `render` (PNG). |
| `disclaimer` | texto fijo en español; va en **cada** respuesta. |

Errores:

| Código | Cuándo |
|---|---|
| 415 | formato distinto de JPEG, PNG o WebP (GIF, BMP, TIFF, …) |
| 413 | archivo mayor de 10 MB |
| 422 | archivo vacío, ilegible como imagen, o con lado menor de 64 px; también si falta el campo `file` |

Un solo archivo por petición; sin lotes.

### `GET /health`

`{"status": "ok", "model_version": "4e7ab61-e0ecf981"}`. 503 si el paquete no cargó.
Es el `HEALTHCHECK` de la imagen Docker.

### `GET /model-info`

La API se describe a sí misma: `manifest.json` completo, `metrics.json` (métricas del test
con intervalos), `thresholds.json`, `calibration.json`, `preprocess.json` y el `disclaimer`.
Con el paquete actual:

| Métrica (test, 5,252 imágenes, 86 melanomas) | Valor [IC 95 % por paciente] |
|---|---|
| AUC-ROC | 0.834 [0.795, 0.872] |
| AUPRC | 0.096 [0.065, 0.154] |
| Sensibilidad en τ95 | 0.965 (83 de 86) |
| Especificidad en τ95 | 0.428 |
| VPP / VPN en τ95 | 0.027 / 0.9986 |

## Preprocesamiento en dos etapas

Las imágenes de entrenamiento pasaron por dos redimensionados: F1 las bajó a 512 px de lado
largo (PIL, `draft` + Lanczos, JPEG calidad 95) y la cadena de entrenamiento las llevó a 224
(albumentations: `SmallestMaxSize(224, cv2.INTER_CUBIC)` → `CenterCrop(224)` →
`Normalize`). Los usuarios suben originales de hasta 6,000 px, y un solo cúbico sin antialias
de 6,000 a 224 produce aliasing que el modelo nunca vio. `services/api/preprocess.py`
reproduce las dos etapas leyendo todo de `preprocess.json`:

1. **Etapa 1 (F1):** si el lado largo supera `stage1.long_side` (512), `draft` de PIL,
   Lanczos al lado largo 512 sin agrandar y reencodificación JPEG a calidad 95 **en
   memoria** (el modelo vio píxeles recomprimidos). Si la imagen ya mide ≤ 512, no se toca.
   Reproduce `melanoma.data.resize.resize_one` bit a bit (`test_stage1_matches_f1_resize`).
2. **Etapa 2 (entrenamiento):** lado corto a 224 con `cv2.resize(INTER_CUBIC)`, recorte
   central, `(x / 255 − mean) / std`, CHW. Es la misma función que usó albumentations, así
   que la paridad es exacta; por eso `opencv-python-headless` está en el grupo `api`
   (+60 MB de imagen, 152 MB en total, contra el cúbico en numpy que dejaba diferencias de
   logit de hasta 6.5e-3).

Sin la etapa 1, `PIL.Image.resize` directo a 224 (con antialias) daba logits distintos en
hasta ±2. Con las dos etapas, sobre 200 imágenes de 512 px y 100 originales de 640 a 6,000 px
de validación, la diferencia máxima de logit contra la cadena de entrenamiento es 1.7e-5 y
9.3e-6, con decisiones idénticas (`reports/f6_parity.json`).

`tests/test_bundle.py::test_preprocess_from_json_only` verifica que cambiar `mean` en el JSON
cambia la salida y que no hay constantes de normalización en `services/api/`.

## Qué esperar

Con τ95 la API devuelve `refer: true` para la mayoría de las imágenes benignas (especificidad
0.43 en el test). No es un error: es el punto de operación de alta sensibilidad de F4. La
interfaz (F7) debe mostrar la probabilidad y el contexto, no un semáforo.

Latencia en CPU de laptop (`reports/f6_latency.md`): p50 ≈ 35 ms, p95 ≈ 45 ms con una imagen
de 512 px; p50 ≈ 50 ms, p95 ≈ 60 ms con un original de 1872 px (`reports/f6_latency_original.md`).
