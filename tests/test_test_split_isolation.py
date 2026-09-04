"""Prueba estructural: ningún módulo bajo src/melanoma/ referencia el split de prueba.

El único archivo autorizado a leer ``data/splits/test.txt`` es ``scripts/evaluate_test.py``
(F4), con un presupuesto de dos accesos en todo el proyecto (ver README). El guardarraíl se
pone antes de que exista la tentación, igual que ``test_api_dependency_isolation``.
"""

from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN = re.compile(r"test\.txt|[\"']test[\"']\s*\)?\s*\.txt|splits?_dir\s*/\s*[\"']test")


def test_test_split_isolation(root_dir: Path) -> None:
    offenders = []
    for path in sorted((root_dir / "src" / "melanoma").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if FORBIDDEN.search(line):
                offenders.append(f"{path.relative_to(root_dir)}:{lineno}: {line.strip()}")
    assert not offenders, "src/ referencia el split de prueba:\n" + "\n".join(offenders)


def test_only_evaluate_script_may_read_test_split(root_dir: Path) -> None:
    """En scripts/, solo evaluate_test.py puede leer test.txt; los de F1 solo lo escriben."""
    allowed = {"evaluate_test.py", "make_splits.py", "eda_report.py"}
    offenders = []
    for path in sorted((root_dir / "scripts").rglob("*.py")):
        if path.name in allowed:
            continue
        if re.search(r"test\.txt", path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(root_dir)))
    assert not offenders, f"scripts no autorizados referencian test.txt: {offenders}"
