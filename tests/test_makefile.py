"""Todos los objetivos del Makefile son comandos: deben estar en .PHONY. Sin eso, un archivo o
directorio con el mismo nombre (kaggle/, data/, reports/...) hace que make responda
"is up to date" y no ejecute nada, como pasó con `make kaggle` el 2026-09-17."""

from __future__ import annotations

import re
from pathlib import Path


def _makefile(root_dir: Path) -> tuple[list[str], set[str]]:
    text = (root_dir / "Makefile").read_text(encoding="utf-8")
    targets = re.findall(r"^([A-Za-z][A-Za-z0-9_-]*):", text, flags=re.M)
    phony: set[str] = set()
    for m in re.finditer(r"^\.PHONY:((?:.*\\\n)*.*)$", text, flags=re.M):
        phony |= set(m.group(1).replace("\\\n", " ").split())
    return targets, phony


def test_every_target_is_phony(root_dir: Path) -> None:
    targets, phony = _makefile(root_dir)
    assert targets, "no se encontraron objetivos"
    missing = [t for t in targets if t not in phony]
    assert not missing, f"objetivos fuera de .PHONY: {missing}"


def test_targets_colliding_with_paths_are_phony(root_dir: Path) -> None:
    """Un objetivo que coincide con un archivo o carpeta del repo solo es seguro si es .PHONY."""
    targets, phony = _makefile(root_dir)
    colliding = [t for t in targets if (root_dir / t).exists()]
    unsafe = [t for t in colliding if t not in phony]
    assert not unsafe, f"objetivos que colisionan con rutas del repo y no son .PHONY: {unsafe}"
