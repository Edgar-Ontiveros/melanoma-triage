# ruff: noqa: E501
"""F6.4 — Latencia en CPU: N peticiones secuenciales a POST /predict con una imagen de
validación (lote 1), contra la API en proceso (TestClient) y, opcionalmente, contra un
servidor levantado (--url). Reporta p50/p95/p99 del total y el desglose por etapa que
devuelve la propia API (decode, preprocess, onnx, cam, render).

Uso: uv run python scripts/bench_api.py [--bundle models/bundle] [--n 200] [--threads 4]
                                          [--image data/processed/isic2020_512/ISIC_xxx.jpg]
                                          [--url http://127.0.0.1:8000]
→ reports/f6_latency.md
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "ISIC_0096201"  # benigna, la del panel de artefactos de F5


def quantiles(values: list[float]) -> dict[str, float]:
    v = np.asarray(values)
    return {
        "p50": float(np.quantile(v, 0.5)),
        "p95": float(np.quantile(v, 0.95)),
        "p99": float(np.quantile(v, 0.99)),
        "mean": float(v.mean()),
        "max": float(v.max()),
    }


def run_in_process(
    bundle: Path, threads: int, data: bytes, n: int
) -> tuple[list[float], dict[str, list[float]], dict]:
    from fastapi.testclient import TestClient
    from services.api.main import build_app

    with TestClient(build_app(bundle, threads)) as client:
        totals: list[float] = []
        stages: dict[str, list[float]] = {}
        first = None
        for _ in range(5):  # calentamiento
            client.post("/api/predict", files={"file": ("x.jpg", data, "image/jpeg")})
        for _ in range(n):
            t0 = time.perf_counter()
            r = client.post("/api/predict", files={"file": ("x.jpg", data, "image/jpeg")})
            totals.append((time.perf_counter() - t0) * 1000)
            body = r.json()
            first = first or body
            for k, v in body["timings_ms"].items():
                stages.setdefault(k, []).append(v)
            stages.setdefault("api_total", []).append(body["latency_ms"])
        return totals, stages, first


def run_http(url: str, data: bytes, n: int) -> tuple[list[float], dict[str, list[float]], dict]:
    import httpx

    totals, stages, first = [], {}, None
    with httpx.Client(base_url=url, timeout=30) as client:
        for _ in range(5):
            client.post("/api/predict", files={"file": ("x.jpg", data, "image/jpeg")})
        for _ in range(n):
            t0 = time.perf_counter()
            r = client.post("/api/predict", files={"file": ("x.jpg", data, "image/jpeg")})
            totals.append((time.perf_counter() - t0) * 1000)
            body = r.json()
            first = first or body
            for k, v in body["timings_ms"].items():
                stages.setdefault(k, []).append(v)
            stages.setdefault("api_total", []).append(body["latency_ms"])
    return totals, stages, first


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="models/bundle")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--image", default=f"data/processed/isic2020_512/{DEFAULT_IMAGE}.jpg")
    ap.add_argument("--url", default=None)
    ap.add_argument("--out", default="reports/f6_latency.md")
    a = ap.parse_args()
    image = ROOT / a.image
    data = image.read_bytes()
    if a.url:
        totals, stages, first = run_http(a.url, data, a.n)
        mode = f"HTTP contra {a.url}"
    else:
        totals, stages, first = run_in_process(ROOT / a.bundle, a.threads, data, a.n)
        mode = f"en proceso (TestClient), {a.threads} hilos intra-op"
    q = quantiles(totals)
    rows = "\n".join(
        f"| {k} | {quantiles(v)['p50']:.1f} | {quantiles(v)['p95']:.1f} | {quantiles(v)['p99']:.1f} | {quantiles(v)['mean']:.1f} |"
        for k, v in stages.items()
    )
    cpu = platform.processor() or platform.machine()
    try:
        model_name = [
            ln.split(":", 1)[1].strip()
            for ln in Path("/proc/cpuinfo").read_text().splitlines()
            if ln.startswith("model name")
        ][0]
    except Exception:  # noqa: BLE001
        model_name = cpu
    verdict = (
        "**cumple** el objetivo p95 < 500 ms"
        if q["p95"] < 500
        else "**NO cumple** el objetivo p95 < 500 ms"
    )
    md = f"""# F6 — Latencia de la API en CPU

_Generado por `scripts/bench_api.py` el {time.strftime("%Y-%m-%d")}._

- Máquina: {model_name}, {os.cpu_count()} hilos lógicos, {platform.system()} {platform.release()}.
- Modo: {mode}. Modelo `{first["model_version"]}`; imagen `{image.name}` ({first["input"]["width"]}×{first["input"]["height"]} {first["input"]["format"]}, {len(data) / 1024:.0f} KB).
- {a.n} peticiones secuenciales a `POST /predict`, lote 1, tras 5 de calentamiento.

## Latencia total por petición (cliente, ms)

| p50 | p95 | p99 | media | máx |
|--:|--:|--:|--:|--:|
| {q["p50"]:.1f} | {q["p95"]:.1f} | {q["p99"]:.1f} | {q["mean"]:.1f} | {q["max"]:.1f} |

La API {verdict}.

## Desglose por etapa (medido dentro de la API, ms)

| etapa | p50 | p95 | p99 | media |
|:--|--:|--:|--:|--:|
{rows}

`api_total` es el tiempo dentro del endpoint (desde recibir los bytes hasta serializar); la
diferencia con la latencia del cliente es el transporte y el análisis del multipart.
`render` es el PNG del CAM superpuesto (base64); no corre cuando el mapa es nulo.
"""
    out = ROOT / a.out
    out.write_text(md, encoding="utf-8")
    (out.with_suffix(".json")).write_text(
        json.dumps(
            {
                "mode": mode,
                "n": a.n,
                "total": q,
                "stages": {k: quantiles(v) for k, v in stages.items()},
            },
            indent=2,
        )
    )
    print(
        f"p50 {q['p50']:.1f} ms · p95 {q['p95']:.1f} ms · p99 {q['p99']:.1f} ms → {out.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
