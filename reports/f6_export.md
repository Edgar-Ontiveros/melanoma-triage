# F6 — Exportación y API: resumen

_2026-09-21. Fuentes: `reports/f6_parity.json`, `reports/f6_latency.md`, `models/bundle/manifest.json`, salida de `docker build`. Contrato en `docs/API.md`._

## Paquete de modelo (`make export-bundle`)

| | |
|---|---|
| `model_version` | `6c330a8-e0ecf981` (git SHA corto + 8 hex del SHA256 del checkpoint `e0ecf981…4710040`, verificado contra `reports/f3_final_model.json`) |
| ONNX | opset 17 (exportador TorchScript; el de `torch.export` no puede producir 17: genera 18 y el conversor falla en `Pad`), lote dinámico, salidas `logit [N,1]` y `features [N,1280,7,7]`; `onnx.checker` y una pasada en onnxruntime antes de guardar |
| Tamaño | 80.6 MB (siete archivos); tarball del release 74.9 MB |
| Versiones | torch 2.14.0+cpu, onnx 1.23.0, onnxruntime 1.29.0, Python 3.12 |

## Paridad con PyTorch (`make verify-bundle`, 200 imágenes de validación, semilla 20260904)

| Comparación | Criterio | Resultado |
|---|---|---|
| Logit ONNX vs torch (mismo tensor) | máx \|Δ\| < 1e-4 | **1.8e-5** ✔ |
| Platt numpy (API) vs `melanoma.eval.Platt` | máx \|Δ\| < 1e-6 | **2.8e-17** ✔ |
| CAM desde `features` ONNX × `cam_weights.npy` vs `melanoma.explain` | correlación > 0.999 | **mín 0.99999999996** en 193; 7 con ambos mapas nulos ✔ |
| Decisión en τ95 | idéntica en las 200 | **0 diferencias** ✔ |
| Cadena completa (JPEG → PIL → numpy → ONNX → Platt) vs entrenamiento (albumentations → torch) | máx \|Δ logit\| < 1e-3 | **máx 6.5e-3, media 6.4e-4, p95 2.6e-3; decisiones idénticas en las 200** ✘ en el máximo |

La única desviación es el máximo de la cadena completa: el redimensionado cúbico en numpy
coincide con `cv2.INTER_CUBIC` en todos los píxeles salvo ~1 de cada 100,000 (redondeo ±1,
508 píxeles de 47 millones sobre las 200 imágenes), y esos píxeles mueven el logit hasta
6.5e-3 en la peor imagen. `PIL.Image.resize` (antialias) habría dado diferencias de ±2 en el
logit. Ninguna decisión cambia. Se reporta tal cual; no se relajó la tolerancia.

## API (`services/api/`)

`POST /predict`, `GET /health`, `GET /model-info`; `disclaimer` en cada respuesta; caso
`no_positive_evidence` con `grid` de ceros y `png_base64: null`; 415 / 413 / 422 según el
error; el paquete se verifica por SHA256 al arrancar. Dependencias: solo el grupo `api`.

## Latencia en CPU (`make bench-api`, 200 peticiones, lote 1, 4 hilos intra-op)

| p50 | p95 | p99 | objetivo |
|--:|--:|--:|:--|
| 50.9 ms | 59.8 ms | 66.0 ms | p95 < 500 ms ✔ |

Desglose típico: ONNX ≈ 34 ms, preprocesamiento numpy ≈ 20 ms, PNG del CAM ≈ 9 ms,
decodificación ≈ 8 ms (ver `reports/f6_latency.md`).

## Docker (`make docker-api`)

Imagen `melanoma-api`: **92.0 MB** (límite 400 MB; la spec esperaba 250–320). Multietapa
sobre `python:3.12-slim-bookworm`, usuario sin privilegios, `HEALTHCHECK` contra `/health`,
paquete montado en `/bundle` (no va dentro de la imagen). CI construye la imagen, la levanta
con el paquete de prueba y verifica `/health` y `/predict`.

## Release

Paquete real publicado como release asset de GitHub (`melanoma-bundle-6c330a8-e0ecf981.tar.gz`,
SHA256 en la descripción del release; ver `docs/specs/F6.md`, sección «Estado»).
