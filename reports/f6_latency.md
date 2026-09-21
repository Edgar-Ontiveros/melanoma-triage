# F6 — Latencia de la API en CPU

_Generado por `scripts/bench_api.py` el 2026-09-21._

- Máquina: AMD Ryzen 7 250 w/ Radeon 780M Graphics, 16 hilos lógicos, Linux 6.18.33.2-microsoft-standard-WSL2.
- Modo: en proceso (TestClient), 4 hilos intra-op. Modelo `6c330a8-e0ecf981`; imagen `ISIC_0096201.jpg` (512×288 JPEG, 47 KB).
- 200 peticiones secuenciales a `POST /predict`, lote 1, tras 5 de calentamiento.

## Latencia total por petición (cliente, ms)

| p50 | p95 | p99 | media | máx |
|--:|--:|--:|--:|--:|
| 50.9 | 59.8 | 66.0 | 51.9 | 74.3 |

La API **cumple** el objetivo p95 < 500 ms.

## Desglose por etapa (medido dentro de la API, ms)

| etapa | p50 | p95 | p99 | media |
|:--|--:|--:|--:|--:|
| decode | 1.0 | 1.4 | 2.2 | 1.1 |
| preprocess | 15.8 | 19.2 | 20.8 | 16.3 |
| onnx | 23.9 | 32.3 | 35.4 | 25.1 |
| cam | 0.1 | 0.2 | 0.2 | 0.1 |
| render | 7.0 | 8.7 | 9.8 | 7.2 |
| api_total | 49.1 | 58.0 | 63.7 | 49.9 |

`api_total` es el tiempo dentro del endpoint (desde recibir los bytes hasta serializar); la
diferencia con la latencia del cliente es el transporte y el análisis del multipart.
`render` es el PNG del CAM superpuesto (base64); no corre cuando el mapa es nulo.
