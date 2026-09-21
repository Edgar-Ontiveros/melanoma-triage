# F6 — Exportación y API: resumen

_2026-09-21 (actualizado tras el ajuste de dos etapas). Fuentes: `reports/f6_parity.json`, `reports/f6_latency.md`, `reports/f6_latency_original.md`, `models/bundle/manifest.json`, salida de `docker build`. Contrato en `docs/API.md`._

## Paquete de modelo (`make export-bundle`)

| | |
|---|---|
| `model_version` | `4e7ab61-e0ecf981` (git SHA corto + 8 hex del SHA256 del checkpoint `e0ecf981…4710040`, verificado contra `reports/f3_final_model.json`) |
| ONNX | opset 17 (exportador TorchScript; el de `torch.export` no puede producir 17: genera 18 y el conversor falla en `Pad`), lote dinámico, salidas `logit [N,1]` y `features [N,1280,7,7]`; `onnx.checker` y una pasada en onnxruntime antes de guardar |
| `preprocess.json` | `stage1` = redimensionado de F1 (lado largo 512, Lanczos, JPEG 95, `draft`; de `configs/data/isic2020.yaml`) + etapa 2 de timm (224, bicúbico, media/desv 0.5) |
| Tamaño | 80.6 MB (siete archivos); tarball del release 74.9 MB |
| Versiones | torch 2.14.0+cpu, onnx 1.23.0, onnxruntime 1.29.0, OpenCV 5.0.0, Python 3.12 |

## Paridad con PyTorch (`make verify-bundle`, semilla 20260904)

| Comparación | Criterio | Resultado |
|---|---|---|
| Logit ONNX vs torch (mismo tensor, 200 val) | máx \|Δ\| < 1e-4 | **1.8e-5** ✔ |
| Platt numpy (API) vs `melanoma.eval.Platt` | máx \|Δ\| < 1e-6 | **2.8e-17** ✔ |
| CAM desde `features` ONNX × `cam_weights.npy` vs `melanoma.explain` | correlación > 0.999 | **mín 0.99999999996** en 193; 7 con ambos mapas nulos ✔ |
| Decisión en τ95 | idéntica en las 200 | **0 diferencias** ✔ |
| Cadena completa desde JPEG de **512 px** (200 val) vs entrenamiento | máx \|Δ logit\| < 1e-3 | **máx 1.7e-5, media 3.1e-6; decisiones idénticas** ✔ |
| Cadena completa desde **originales** de `data/raw` (100 val, lado largo 640–6,000 px) vs `resize_one` de F1 + entrenamiento | máx \|Δ logit\| < 1e-3 | **máx 9.3e-6, media 2.4e-6; decisiones idénticas** ✔ |

`all_pass = true`. Antes del ajuste, con el cúbico en numpy, la cadena a 512 px daba máx
6.5e-3 (1 de cada 100,000 píxeles redondeado distinto de OpenCV) y no había prueba con
originales. Ahora la etapa 2 usa la misma `cv2.resize` que albumentations (paridad exacta) y
la etapa 1 replica bit a bit `melanoma.data.resize.resize_one`, incluida la reencodificación
JPEG a calidad 95. El original de 1872 × 1053 px y su versión de 512 px dan la misma
probabilidad (0.015491).

## API (`services/api/`)

`POST /predict`, `GET /health`, `GET /model-info`; `disclaimer` en cada respuesta; caso
`no_positive_evidence` con `grid` de ceros y `png_base64: null`; 415 / 413 / 422 según el
error; el paquete se verifica por SHA256 al arrancar. Dependencias: solo el grupo `api`
(incluye `opencv-python-headless` desde el ajuste).

## Latencia en CPU (`make bench-api`, 200 peticiones, lote 1, 4 hilos intra-op)

| imagen | p50 | p95 | p99 | objetivo |
|:--|--:|--:|--:|:--|
| 512 × 288 px (`data/processed`) | 35.2 ms | 45.2 ms | 52.0 ms | p95 < 500 ms ✔ |
| original 1872 × 1053 px (`data/raw`) | 50.1 ms | 60.0 ms | 64.6 ms | ✔ |

Desglose típico con el original: decodificación + etapa 1 ≈ 10 ms, etapa 2 (cv2) ≈ 1 ms,
ONNX ≈ 30 ms, PNG del CAM ≈ 7 ms. Con el cúbico en numpy la etapa 2 costaba ≈ 20 ms.

## Docker (`make docker-api`)

Imagen `melanoma-api`: **152.1 MB** (límite 400; era 92.0 MB antes de agregar OpenCV, que
cuesta ≈ 60 MB). Multietapa sobre `python:3.12-slim-bookworm`, usuario sin privilegios,
`HEALTHCHECK` contra `/health`, paquete montado en `/bundle` (no va dentro de la imagen). CI
construye la imagen, la levanta con el paquete de prueba y verifica `/health` y `/predict`.

## Release

Paquete real publicado como release asset de GitHub: `model-4e7ab61-e0ecf981`,
`melanoma-bundle-4e7ab61-e0ecf981.tar.gz`, SHA256
`a884adac90fa6924ee090702d728606c30ecbfadf3b9efab4583064dbd38465d` (en la descripción del
release y como archivo `.sha256`). El release anterior (`model-6c330a8-e0ecf981`) se retiró.
