"""F1.5 — Redimensiona a lado largo ``data.resize.long_side`` px, JPEG ``jpeg_quality``.

Escribe en ``data.resize.out_dir`` y agrega al manifiesto ``sha256_resized`` y
``relpath_resized`` (más ``width_resized``, ``height_resized``, ``bytes_resized``).
Reporta el tamaño total en disco en ``reports/resize_report.md``.

Uso: ``uv run python scripts/resize_images.py``
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
from omegaconf import DictConfig

from melanoma.data.manifest import read_manifest
from melanoma.data.resize import resize_many


@hydra.main(config_path="../configs", config_name="prepare", version_base="1.3")
def main(cfg: DictConfig) -> None:
    d = cfg.data
    root = Path(d.root)
    out_dir = Path(d.resize.out_dir)
    m = read_manifest(d.manifest_path)

    pairs = [
        (root / rp, out_dir / f"{image_id}.jpg")
        for image_id, rp in zip(m["image_id"], m["relpath_original"], strict=True)
    ]
    t0 = time.time()
    results = resize_many(pairs, d.resize.long_side, d.resize.jpeg_quality, d.resize.workers)
    elapsed = time.time() - t0

    m["sha256_resized"] = [r["sha256"] for r in results]
    m["relpath_resized"] = [str(dst.relative_to(root)) for _, dst in pairs]
    m["width_resized"] = [r["width"] for r in results]
    m["height_resized"] = [r["height"] for r in results]
    m["bytes_resized"] = [r["bytes"] for r in results]
    m.to_csv(d.manifest_path, index=False)

    total = int(m["bytes_resized"].sum())
    orig = int(m["bytes"].sum())
    not_downscaled = int((m[["width", "height"]].max(axis=1) <= d.resize.long_side).sum())
    long_out = m[["width_resized", "height_resized"]].max(axis=1)
    lines = [
        "# Redimensionado ISIC 2020",
        "",
        f"Generado por `scripts/resize_images.py`. Lado largo {d.resize.long_side} px, "
        f"relación de aspecto preservada, JPEG calidad {d.resize.jpeg_quality}, "
        f"remuestreo Lanczos. Salida: `{d.resize.out_dir}`.",
        "",
        f"- Imágenes: {len(m):,}",
        f"- Tamaño total en disco: **{total / 1e9:.2f} GB** (originales: {orig / 1e9:.2f} GB, "
        f"factor {orig / total:.1f}x)",
        f"- Imágenes que ya eran ≤ {d.resize.long_side} px y no se redujeron: {not_downscaled}",
        f"- Dimensiones resultantes: lado largo mín {int(long_out.min())}, "
        f"máx {int(long_out.max())}",
        f"- Tiempo: {elapsed / 60:.1f} min con {d.resize.workers} procesos",
        "",
        "Columnas agregadas al manifiesto: `sha256_resized`, `relpath_resized`, "
        "`width_resized`, `height_resized`, `bytes_resized`.",
        "",
    ]
    report = Path(d.reports_dir) / "resize_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
