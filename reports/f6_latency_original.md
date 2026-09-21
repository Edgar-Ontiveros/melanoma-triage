# F6 — Latencia de la API en CPU

_Generado por `scripts/bench_api.py` el 2026-09-21._

- Máquina: AMD Ryzen 7 250 w/ Radeon 780M Graphics, 16 hilos lógicos, Linux 6.18.33.2-microsoft-standard-WSL2.
- Modo: en proceso (TestClient), 4 hilos intra-op. Modelo `4e7ab61-e0ecf981`; imagen `ISIC_0096201.jpg` (1872×1053 JPEG, 158 KB).
- 200 peticiones secuenciales a `POST /predict`, lote 1, tras 5 de calentamiento.

## Latencia total por petición (cliente, ms)

| p50 | p95 | p99 | media | máx |
|--:|--:|--:|--:|--:|
| 50.1 | 60.0 | 64.6 | 50.9 | 71.9 |

La API **cumple** el objetivo p95 < 500 ms.

## Desglose por etapa (medido dentro de la API, ms)

| etapa | p50 | p95 | p99 | media |
|:--|--:|--:|--:|--:|
| decode_stage1 | 12.2 | 16.0 | 17.9 | 12.6 |
| preprocess | 1.1 | 1.4 | 1.9 | 1.1 |
| onnx | 26.3 | 35.7 | 39.6 | 27.3 |
| cam | 0.1 | 0.2 | 0.3 | 0.1 |
| render | 7.3 | 9.1 | 9.6 | 7.5 |
| api_total | 47.9 | 57.8 | 62.5 | 48.8 |

`api_total` es el tiempo dentro del endpoint (desde recibir los bytes hasta serializar); la
diferencia con la latencia del cliente es el transporte y el análisis del multipart.
`render` es el PNG del CAM superpuesto (base64); no corre cuando el mapa es nulo.
