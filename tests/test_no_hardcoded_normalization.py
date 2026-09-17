"""Criterio 4 de F2: ningún valor de normalización escrito a mano en src/ (verificable por grep).

La media y desviación de ImageNet (0.485, 0.456, 0.406 / 0.229, 0.224, 0.225), las de
Inception (0.5 / 0.5) o cualquier `mean=(...)` literal solo pueden venir del data_config
que devuelve la fábrica.
"""

from __future__ import annotations

import re
from pathlib import Path

PATTERNS = [
    r"0\.485|0\.456|0\.406|0\.229|0\.224|0\.225",
    r"IMAGENET_DEFAULT_MEAN|IMAGENET_DEFAULT_STD|IMAGENET_INCEPTION",
    r"\bmean\s*=\s*[\[(]\s*0\.",
    r"\bstd\s*=\s*[\[(]\s*0\.",
    r"Normalize\(\s*[\[(]",
]


def test_no_hardcoded_normalization(root_dir: Path) -> None:
    offenders = []
    for path in sorted((root_dir / "src" / "melanoma").rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for pat in PATTERNS:
                if re.search(pat, line):
                    offenders.append(f"{path.relative_to(root_dir)}:{lineno}: {line.strip()}")
    assert not offenders, "normalización escrita a mano en src/:\n" + "\n".join(offenders)


def test_grep_style_check_for_scripts(root_dir: Path) -> None:
    """Los scripts tampoco pueden llevar la media/desviación de ImageNet."""
    offenders = []
    for path in sorted((root_dir / "scripts").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if re.search(PATTERNS[0], text):
            offenders.append(str(path.relative_to(root_dir)))
    assert not offenders, f"scripts con normalización a mano: {offenders}"
