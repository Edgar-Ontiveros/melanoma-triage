"""Guardarraíles estáticos para los notebooks de Kaggle y para el grupo `core`.

La verificación completa (venv limpio, clon, ejecución celda a celda) es
``scripts/check_notebooks.py`` (``make check-notebooks``) y no corre en CI por su costo. Aquí se
comprueba lo barato: que cada ``import`` de ``melanoma.*`` en los notebooks exista de verdad, que
el `src/` entre en `sys.path` antes del primer import (el .pth del editable no se lee en un kernel
ya corriendo) y que los módulos de F1 se importen sin torch.
"""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

NOTEBOOKS = sorted(Path(__file__).resolve().parents[1].glob("notebooks/*.ipynb"))


def _code_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _melanoma_imports(src: str) -> list[tuple[str, str | None]]:
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("melanoma"):
            out += [(node.module, a.name) for a in node.names]
        elif isinstance(node, ast.Import):
            out += [(a.name, None) for a in node.names if a.name.startswith("melanoma")]
    return out


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_cells_parse_and_imports_exist(notebook: Path) -> None:
    cells = _code_cells(notebook)
    assert cells, f"{notebook.name} no tiene celdas de código"
    for i, src in enumerate(cells):
        ast.parse(src)  # sin magics ni sintaxis inválida
        for module, name in _melanoma_imports(src):
            mod = importlib.import_module(module)
            assert name is None or hasattr(mod, name), f"{notebook.name} celda {i}: {module}.{name}"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_src_on_sys_path_before_first_melanoma_import(notebook: Path) -> None:
    cells = _code_cells(notebook)
    first_import = None
    for i, src in enumerate(cells):
        for node in ast.walk(ast.parse(src)):
            is_from = isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "melanoma"
            )
            is_imp = isinstance(node, ast.Import) and any(
                a.name.startswith("melanoma") for a in node.names
            )
            if is_from or is_imp:
                first_import = min(first_import or (i, node.lineno), (i, node.lineno))
    if first_import is None:
        pytest.skip("el notebook no importa melanoma")
    marker = 'sys.path.insert(0, f"{REPO_DIR}/src")'
    marker_at = next(
        (
            (i, n)
            for i, src in enumerate(cells)
            for n, line in enumerate(src.splitlines(), 1)
            if marker in line
        ),
        None,
    )
    assert marker_at is not None, f"{notebook.name}: falta {marker}"
    assert marker_at < first_import, f"{notebook.name}: src/ se agrega después del primer import"


def test_core_modules_import_without_torch() -> None:
    """Los módulos de F1 (los que usa kaggle_prepare_512) no deben cargar torch ni lightning."""
    code = (
        "import sys\n"
        "for m in ('torch', 'lightning', 'albumentations', 'timm', 'torchmetrics'):\n"
        "    sys.modules[m] = None\n"
        "import melanoma.data\n"
        "from melanoma.data.resize import resize_many\n"
        "from melanoma.data.manifest import read_manifest\n"
        "from melanoma.data.paths import resolve_paths\n"
        "print('core ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
