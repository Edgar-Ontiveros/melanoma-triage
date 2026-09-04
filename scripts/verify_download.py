"""F1.1 — Verifica la descarga oficial de ISIC 2020 y extrae las imágenes.

1. Comprueba que el ZIP tiene exactamente los bytes que reporta S3 (``data.source.zip_bytes``).
2. Extrae los JPEG a ``data.images_dir`` si aún no están.
3. Cuenta imágenes en disco y contrasta con el CSV oficial y con ``data.expected``.

Uso: ``uv run python scripts/verify_download.py``
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from melanoma.data.manifest import load_ground_truth, verify_official_counts


def extract(zip_path: Path, images_dir: Path) -> None:
    """Extrae los ``*.jpg`` del ZIP directamente en ``images_dir`` (aplanando carpetas)."""
    images_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if m.filename.lower().endswith(".jpg")]
        for m in tqdm(members, desc="extrayendo", unit="img"):
            target = images_dir / Path(m.filename).name
            if target.exists() and target.stat().st_size == m.file_size:
                continue
            with zf.open(m) as src, open(target, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)


@hydra.main(config_path="../configs", config_name="prepare", version_base="1.3")
def main(cfg: DictConfig) -> None:
    d = cfg.data
    zip_path = Path(d.images_zip)
    images_dir = Path(d.images_dir)

    if zip_path.exists():
        size = zip_path.stat().st_size
        print(f"ZIP: {zip_path} → {size:,} bytes (esperado {d.source.zip_bytes:,})")
        if size != d.source.zip_bytes:
            sys.exit("ERROR: el tamaño del ZIP no coincide con Content-Length de S3")
    elif not images_dir.exists():
        sys.exit(f"ERROR: no existe {zip_path} ni {images_dir}")

    n_disk = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
    if n_disk < d.expected.images and zip_path.exists():
        extract(zip_path, images_dir)
        n_disk = len(list(images_dir.glob("*.jpg")))
    print(f"Imágenes en disco: {n_disk:,}")

    gt = load_ground_truth(d.ground_truth_csv)
    table = verify_official_counts(gt, dict(d.expected))
    print(table.to_string(index=False))
    ok = bool(table["ok"].all()) and n_disk == d.expected.images
    if not ok:
        sys.exit("ERROR: alguna verificación oficial no cuadra. Fase detenida (spec F1.1).")
    print("F1.1 OK: conteos oficiales verificados.")


if __name__ == "__main__":
    main()
