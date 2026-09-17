"""Parámetros de un notebook: la primera celda de código es la celda de parámetros (convención
de papermill). Se leen con ``ast`` y se sustituyen por texto, línea a línea, sin ejecutar nada.

Lo usan ``scripts/check_notebooks.py`` (sandbox local) y ``scripts/kaggle_run.py`` (kernels por
API) para lanzar el mismo notebook con distintos REPO_SHA, OVERRIDES, rutas, etc.
"""

from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path
from typing import Any


def load_notebook(path: Path | str) -> dict[str, Any]:
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if isinstance(cell["source"], list):
            cell["source"] = "".join(cell["source"])
    return nb


def save_notebook(nb: dict[str, Any], path: Path | str) -> None:
    Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def parameters_cell(nb: dict[str, Any]) -> dict[str, Any]:
    """Primera celda de código."""
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            return cell
    raise ValueError("el notebook no tiene celdas de código")


def read_parameters(nb: dict[str, Any]) -> dict[str, Any]:
    """``{NOMBRE: valor}`` de las asignaciones ``NOMBRE = literal`` de la celda de parámetros."""
    out: dict[str, Any] = {}
    for node in ast.parse(parameters_cell(nb)["source"]).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    out[target.id] = ast.unparse(node.value)
    return out


def substitute_parameters(nb: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Copia del notebook con ``NOMBRE = ...`` reemplazado por ``NOMBRE = repr(valor)``.

    Falla si un nombre no existe en la celda de parámetros: un parámetro mal escrito no debe
    pasar en silencio.
    """
    nb = copy.deepcopy(nb)
    cell = parameters_cell(nb)
    src = cell["source"]
    for name, value in params.items():
        src, n = re.subn(rf"^{re.escape(name)} = .*$", f"{name} = {value!r}", src, flags=re.M)
        if n != 1:
            raise KeyError(f"parámetro {name!r} no está (una vez) en la celda de parámetros")
    cell["source"] = src
    return nb
