"""F7: la interfaz. Sin constantes del modelo en el frontend, limitaciones presentes, ejemplos
de validación (nunca DDI), prefijo /api y SPA, y la imagen Docker con el build."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from services.api.main import build_app, count_sections

from tests.test_bundle import test_bundle  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "services" / "web"
LIMITATIONS = ROOT / "docs" / "f4_limitations.md"


def _web_sources() -> list[Path]:
    return [
        p
        for p in sorted((WEB / "src").rglob("*"))
        if p.is_file() and p.suffix in (".ts", ".tsx", ".js", ".json") and ".test." not in p.name
    ]


def test_web_no_model_constants() -> None:
    """Ningún número de umbral, prevalencia o métrica en services/web/src (fuera de pruebas y
    CSS): ni los literales conocidos ni flotantes con tres o más decimales."""
    known = re.compile(r"0\.0039|0\.965|0\.428|0\.834|0\.0164|0\.096")
    floats = re.compile(r"(?<![\w.])\d+\.\d{3,}(?![\w.])")
    offenders = []
    for path in _web_sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("//", 1)[0]
            if known.search(code) or floats.search(code):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "constantes del modelo en el frontend:\n" + "\n".join(offenders)


def test_limitations_present() -> None:
    """docs/f4_limitations.md existe y tiene al menos 8 secciones o párrafos. La interfaz no
    se cierra sin las limitaciones; las escribe el autor (F4, sección 10)."""
    assert LIMITATIONS.exists(), (
        "falta docs/f4_limitations.md: la ficha del modelo muestra «pendiente» hasta que exista"
    )
    text = LIMITATIONS.read_text(encoding="utf-8")
    assert count_sections(text) >= 8, f"solo {count_sections(text)} bloques en f4_limitations.md"


def test_example_images_are_val_not_ddi() -> None:
    examples = json.loads((WEB / "public" / "examples" / "examples.json").read_text())
    ids = {e["image_id"] for e in examples["items"]}
    assert len(ids) == 4
    val = set((ROOT / "data" / "splits" / "val.txt").read_text().split())
    assert ids <= val, ids - val
    truths = [e["truth"] for e in examples["items"]]
    assert truths.count("melanoma") == 2 and truths.count("benigna") == 2
    for e in examples["items"]:
        assert (WEB / "public" / e["file"]).exists(), e["file"]
    ddi = re.compile(r"raw/ddi|ddi_metadata|\bDDI\b(?!, Stanford\)|\)| \(Stanford)")
    offenders = []
    for path in sorted(WEB.rglob("*")):
        if path.is_file() and "node_modules" not in path.parts and "dist" not in path.parts:
            if path.suffix in (".ts", ".tsx", ".json", ".html", ".css", ".md"):
                text = path.read_text(encoding="utf-8")
                if re.search(r"raw/ddi|ddi_metadata|ddi/images", text, re.I):
                    offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"rutas de DDI en services/web: {offenders}"
    assert not ddi.search("")  # el patrón compila


@pytest.fixture()
def spa_client(test_bundle: Path, tmp_path: Path):  # noqa: F811
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><html><body><div id='root'>SPA</div></body></html>"
    )
    (dist / "assets" / "app.js").write_text("console.log('ok')")
    lim = tmp_path / "f4_limitations.md"
    lim.write_text("# Limitaciones\n\n" + "\n\n".join(f"Párrafo {i}." for i in range(8)))
    with TestClient(
        build_app(test_bundle, intra_op_threads=2, web_dist=dist, limitations_md=lim)
    ) as c:
        yield c


def test_api_prefix_and_spa(spa_client: TestClient) -> None:
    assert spa_client.get("/api/health").status_code == 200
    assert spa_client.get("/api/model-info").json()["model_version"]
    r = spa_client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "SPA" in r.text
    assert spa_client.get("/modelo").status_code == 200 and "SPA" in spa_client.get("/modelo").text
    assert spa_client.get("/assets/app.js").status_code == 200
    assert "SPA" in spa_client.get("/../../etc/passwd").text  # nunca sale de dist
    lim = spa_client.get("/api/limitations").json()
    assert lim["n_sections"] == 9 and lim["markdown"].startswith("# Limitaciones")
    assert spa_client.get("/health").status_code == 200  # cae en la SPA: sin API fuera de /api


def test_limitations_404_when_missing(test_bundle: Path, tmp_path: Path) -> None:  # noqa: F811
    with TestClient(
        build_app(test_bundle, intra_op_threads=2, limitations_md=tmp_path / "no.md")
    ) as c:
        r = c.get("/api/limitations")
    assert r.status_code == 404 and "pendiente" in r.json()["detail"]


def test_web_build_in_docker(test_bundle: Path) -> None:  # noqa: F811
    """La imagen contiene el build y sirve / con 200 (y /api/health). Construye la imagen si
    Docker está disponible; si no, se salta."""
    if shutil.which("docker") is None:
        pytest.skip("docker no disponible")
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip("el demonio de docker no responde")
    build = subprocess.run(
        ["docker", "build", "-q", "-f", "services/api/Dockerfile", "-t", "melanoma-api", "."],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr[-2000:]
    # El contenedor corre como usuario sin privilegios y el tmp de pytest es 700: copia legible.
    readable = Path(tempfile.mkdtemp(prefix="melanoma-bundle-"))
    shutil.copytree(test_bundle, readable / "bundle")
    for path in [readable, *readable.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    name = f"melanoma-api-test-{int(time.time())}"
    run = subprocess.run(
        [
            "docker", "run", "-d", "--name", name, "-p", "127.0.0.1:0:8000",
            "-v", f"{readable / 'bundle'}:/bundle:ro", "melanoma-api",
        ],
        capture_output=True,
        text=True,
    )  # fmt: skip
    assert run.returncode == 0, run.stderr
    try:
        port = subprocess.check_output(["docker", "port", name, "8000/tcp"], text=True).strip()
        port = port.splitlines()[0].rsplit(":", 1)[1]
        base = f"http://127.0.0.1:{port}"
        for _ in range(60):
            try:
                if urllib.request.urlopen(f"{base}/api/health", timeout=2).status == 200:
                    break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        else:
            logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
            pytest.fail(f"la API no levantó en el contenedor:\n{logs.stdout}\n{logs.stderr}")
        with urllib.request.urlopen(f"{base}/", timeout=5) as r:
            assert r.status == 200
            html = r.read().decode()
        assert '<div id="root">' in html and "/assets/" in html
        with urllib.request.urlopen(f"{base}/modelo", timeout=5) as r:
            assert r.status == 200
        with urllib.request.urlopen(f"{base}/examples/examples.json", timeout=5) as r:
            assert len(json.load(r)["items"]) == 4
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        shutil.rmtree(readable, ignore_errors=True)
