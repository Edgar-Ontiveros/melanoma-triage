"""Prueba estructural: nada bajo services/api/ importa torch, timm ni lightning.

Dos capas: análisis estático del AST (importaciones directas) e importación real en
un subproceso limpio (importaciones transitivas, vía `sys.modules`).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

FORBIDDEN = ("torch", "timm", "lightning", "torchvision")


def _api_modules(root_dir: Path) -> list[Path]:
    return sorted((root_dir / "services" / "api").rglob("*.py"))


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_api_has_modules(root_dir: Path) -> None:
    assert _api_modules(root_dir), "services/api/ no tiene módulos Python"


def test_api_no_torch_import_static(root_dir: Path) -> None:
    for path in _api_modules(root_dir):
        bad = _imports_of(path) & set(FORBIDDEN)
        assert not bad, f"{path.relative_to(root_dir)} importa {sorted(bad)}"


@pytest.mark.parametrize("path", [p for p in _api_modules(Path(__file__).resolve().parents[1])])
def test_api_no_torch_import_transitive(root_dir: Path, path: Path) -> None:
    rel = path.relative_to(root_dir).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    module = ".".join(parts)
    code = (
        "import importlib, sys\n"
        f"importlib.import_module({module!r})\n"
        f"loaded = sorted(m for m in sys.modules if m.split('.')[0] in {FORBIDDEN!r})\n"
        "print(loaded)\n"
        "sys.exit(1 if loaded else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"{module} carga módulos prohibidos: {result.stdout.strip()}\n{result.stderr}"
    )
