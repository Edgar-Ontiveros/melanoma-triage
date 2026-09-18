"""Guardarraíles de F4: el protocolo debe estar commiteado, la bitácora se escribe antes de
leer el split de prueba, y DDI nunca se acerca a Kaggle."""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "evaluate_test", ROOT / "scripts/evaluate_test.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_protocol_committed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    et = _load()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "f4_protocol.yaml").write_text("a: 1\n")

    def fake(cmd, text=True):
        if "ls-files" in cmd:
            return "docs/f4_protocol.yaml\n"
        if "status" in cmd:
            return fake.dirty
        raise AssertionError(cmd)

    fake.dirty = " M docs/f4_protocol.yaml\n"
    monkeypatch.setattr(et.subprocess, "check_output", fake)
    with pytest.raises(SystemExit, match="sin commitear"):
        et.ensure_protocol_committed(tmp_path)
    fake.dirty = ""
    assert len(et.ensure_protocol_committed(tmp_path)) == 64
    monkeypatch.setattr(et.subprocess, "check_output", lambda cmd, text=True: "")
    with pytest.raises(SystemExit, match="no está en git"):
        et.ensure_protocol_committed(tmp_path)


def test_protocol_is_committed_in_this_repo() -> None:
    """El protocolo real está versionado y su commit precede al del script de evaluación."""

    def first(path: str) -> list[str]:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "log", "--diff-filter=A", "--format=%ct", "--", path],
            text=True,
        )
        return out.strip().splitlines()

    protocol, script = first("docs/f4_protocol.yaml"), first("scripts/evaluate_test.py")
    assert protocol, "docs/f4_protocol.yaml no está en git"
    if script:  # el script puede no estar commiteado aún en el árbol de trabajo
        assert int(protocol[-1]) <= int(script[-1]), (
            "el protocolo debe commitearse antes que el script"
        )


def test_access_logged_before_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Con el sistema de archivos simulado: la bitácora se escribe antes de abrir test.txt."""
    et = _load()
    events: list[str] = []

    class Stop(Exception):
        pass

    monkeypatch.setattr(et, "ensure_protocol_committed", lambda root, protocol=None: "p" * 64)
    monkeypatch.setattr(et, "verify_checkpoint", lambda root, pc: "c" * 64)
    monkeypatch.setattr(et, "append_access_log", lambda *a, **k: events.append("log") or "line")
    monkeypatch.setattr(et, "verify_split_hashes", lambda d: {})

    def fake_read(splits_dir):
        events.append("read")
        raise Stop

    monkeypatch.setattr(et, "read_test_ids", fake_read)
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        f4=SimpleNamespace(protocol="docs/f4_protocol.yaml", access_log="logs/x.log"),
        data={
            "env": "local",
            "paths": {"local": {"images_dir": "i", "manifest_path": "m", "splits_dir": "s"}},
        },
    )
    monkeypatch.setattr(et.OmegaConf, "load", lambda p: SimpleNamespace(model=SimpleNamespace()))
    with pytest.raises(Stop):
        et.run(cfg, tmp_path, "prueba")
    assert events == ["log", "read"], events


def test_only_evaluate_test_reads_the_split_in_the_script() -> None:
    src = (ROOT / "scripts/evaluate_test.py").read_text(encoding="utf-8")
    assert src.index("append_access_log(root") < src.index("read_test_ids(paths"), (
        "el registro debe ir antes de la lectura"
    )


def test_ddi_never_uploaded() -> None:
    """Ningún script ni notebook mezcla rutas de DDI con llamadas a Kaggle."""
    ddi = re.compile(r"raw/ddi|ddi_metadata|DDI", re.I)
    kaggle = re.compile(r"kaggle|KaggleApi|kernels push|datasets create", re.I)
    offenders = []
    for path in sorted(
        list((ROOT / "scripts").glob("*.py")) + list((ROOT / "notebooks").glob("*.ipynb"))
    ):
        text = path.read_text(encoding="utf-8")
        if ddi.search(text) and kaggle.search(text):
            offenders.append(path.name)
    assert not offenders, f"DDI junto a Kaggle en: {offenders}"
