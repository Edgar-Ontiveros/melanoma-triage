"""F1.2 — Genera ``data/manifests/isic2020.csv`` y ``reports/manifest_report.md``.

Una fila por imagen: huella SHA256 del archivo original, ruta relativa, dimensiones,
bytes, metadatos del CSV oficial y etiqueta. Reporta faltantes por columna.

Uso: ``uv run python scripts/build_manifest.py``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig

from melanoma.data.manifest import (
    build_manifest,
    load_ground_truth,
    missing_report,
    verify_official_counts,
)


@hydra.main(config_path="../configs", config_name="prepare", version_base="1.3")
def main(cfg: DictConfig) -> None:
    d = cfg.data
    gt = load_ground_truth(d.ground_truth_csv)
    counts = verify_official_counts(gt, dict(d.expected))
    if not counts["ok"].all():
        print(counts.to_string(index=False))
        sys.exit("ERROR: conteos oficiales no verificados; no se genera el manifiesto.")

    t0 = time.time()
    manifest = build_manifest(gt, d.images_dir, d.root, workers=d.hashing.workers)
    elapsed = time.time() - t0
    out = Path(d.manifest_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    print(f"Manifiesto: {out} ({len(manifest):,} filas, hashing en {elapsed / 60:.1f} min)")

    missing = missing_report(manifest)
    report = Path(d.reports_dir) / "manifest_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = int(manifest["bytes"].sum())
    lines = [
        "# Manifiesto ISIC 2020 — reporte de generación",
        "",
        f"Generado por `scripts/build_manifest.py` a partir de `{d.ground_truth_csv}` y",
        f"`{d.images_dir}`. Salida: `{d.manifest_path}`.",
        "",
        "## Verificación contra la publicación oficial (F1.1)",
        "",
        counts.to_markdown(index=False),
        "",
        "Fuente: DOI 10.34970/2020-ds01; Rotemberg et al., *Sci Data* 8, 34 (2021).",
        "",
        "## Archivos",
        "",
        f"- Imágenes: {len(manifest):,}",
        f"- Tamaño total en disco (originales): {total_bytes / 1e9:.2f} GB",
        f"- Duplicados exactos por SHA256: {int(manifest.duplicated('sha256_original').sum())}",
        "",
        "## Faltantes por columna (F1.2)",
        "",
        missing.to_markdown(index=False),
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(missing.to_string(index=False))
    print(f"Reporte: {report}")


if __name__ == "__main__":
    main()
