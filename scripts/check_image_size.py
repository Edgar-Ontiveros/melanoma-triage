"""Falla con código 1 si una imagen Docker supera el límite en MB.

Uso: ``python scripts/check_image_size.py <imagen> <max_mb>``.
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    image, max_mb = sys.argv[1], float(sys.argv[2])
    out = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Size}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    size_mb = int(out) / (1024 * 1024)
    print(f"{image}: {size_mb:.1f} MB (límite {max_mb:.0f} MB)")
    if size_mb > max_mb:
        print("ERROR: la imagen supera el límite. ¿Se coló torch/timm/lightning en `api`?")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
