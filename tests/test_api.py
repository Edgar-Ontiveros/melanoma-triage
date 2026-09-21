"""F6.3: contrato de la API con el paquete de prueba (pesos aleatorios): /predict, /health,
/model-info, códigos de error, caso sin evidencia positiva y verificación del manifiesto al
arrancar."""

from __future__ import annotations

import base64
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from services.api.bundle import BundleError
from services.api.main import DISCLAIMER, build_app

from tests.test_bundle import negative_bundle, synthetic_rgb, test_bundle  # noqa: F401

REQUIRED = {
    "model_version",
    "probability",
    "refer",
    "threshold",
    "operating_point",
    "cam",
    "input",
    "latency_ms",
    "timings_ms",
    "disclaimer",
}


def encode(img: np.ndarray, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client(test_bundle: Path):  # noqa: F811
    with TestClient(build_app(test_bundle, intra_op_threads=2)) as c:
        yield c


def test_health_and_model_info(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    version = r.json()["model_version"]
    info = client.get("/model-info").json()
    assert info["model_version"] == version
    assert set(info["manifest"]["files"]) >= {"model.onnx", "cam_weights.npy"}
    for key in ("auroc", "auprc"):
        assert {"point", "lo", "hi"} <= set(info["metrics"][key])
    assert "tau_95" in info["thresholds"] and info["thresholds"]["operating"] == "tau_95"
    assert info["disclaimer"] == DISCLAIMER
    assert info["manifest"]["kind"] == "test"  # el paquete de prueba se declara como tal


def test_predict_contract(client: TestClient) -> None:
    r = client.post("/predict", files={"file": ("x.jpg", encode(synthetic_rgb()), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert REQUIRED <= set(body)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["refer"] == (body["probability"] >= body["threshold"])
    assert body["disclaimer"] == DISCLAIMER and "dermatólogo" in body["disclaimer"]
    assert body["input"] == {"width": 768, "height": 512, "format": "JPEG"}
    op = body["operating_point"]
    assert op["sensitivity_target"] == pytest.approx(0.95)
    cam = body["cam"]
    assert cam["status"] in ("ok", "no_positive_evidence")
    assert len(cam["grid"]) == 7 and all(len(row) == 7 for row in cam["grid"])
    flat = [v for row in cam["grid"] for v in row]
    assert min(flat) >= 0.0 and max(flat) <= 1.0
    if cam["status"] == "ok":
        assert max(flat) == pytest.approx(1.0, abs=1e-3)
        png = base64.b64decode(cam["png_base64"])
        with Image.open(io.BytesIO(png)) as im:
            assert im.size == (224, 224) and im.format == "PNG"
    else:
        assert cam["png_base64"] is None
    assert {"decode", "preprocess", "onnx", "cam", "render"} <= set(body["timings_ms"])
    assert "logit" not in body and "prob_raw" not in body  # nunca la probabilidad cruda


def test_predict_accepts_png_and_webp(client: TestClient) -> None:
    for fmt, mime in (("PNG", "image/png"), ("WEBP", "image/webp")):
        r = client.post(
            "/predict", files={"file": ("x", encode(synthetic_rgb(300, 300), fmt), mime)}
        )
        assert r.status_code == 200, (fmt, r.text)
        assert r.json()["input"]["format"] == fmt


def test_predict_no_positive_evidence(negative_bundle: Path) -> None:  # noqa: F811
    with TestClient(build_app(negative_bundle, intra_op_threads=2)) as c:
        body = c.post(
            "/predict", files={"file": ("x.jpg", encode(synthetic_rgb()), "image/jpeg")}
        ).json()
    assert body["cam"]["status"] == "no_positive_evidence"
    assert body["cam"]["png_base64"] is None
    assert not any(v for row in body["cam"]["grid"] for v in row)
    assert body["refer"] is False
    assert "render" not in body["timings_ms"] or body["timings_ms"]["render"] < 1.0


def test_predict_rejects_bad_input(client: TestClient) -> None:
    files = lambda data, name="x.bin", mime="application/octet-stream": {"file": (name, data, mime)}  # noqa: E731
    assert (
        client.post(
            "/predict", files=files(encode(synthetic_rgb(128, 128), "GIF"), "x.gif", "image/gif")
        ).status_code
        == 415
    )
    assert (
        client.post(
            "/predict", files=files(encode(synthetic_rgb(128, 128), "BMP"), "x.bmp", "image/bmp")
        ).status_code
        == 415
    )
    assert client.post("/predict", files=files(b"\x00" * (10 * 1024 * 1024 + 1))).status_code == 413
    assert (
        client.post(
            "/predict", files=files(encode(synthetic_rgb(40, 40), "PNG"), "s.png", "image/png")
        ).status_code
        == 422
    )
    r = client.post("/predict", files=files(b"esto no es una imagen"))
    assert r.status_code == 422 and "imagen" in r.json()["detail"]
    assert client.post("/predict", files=files(b"")).status_code == 422
    assert client.post("/predict").status_code == 422  # sin archivo


def test_app_refuses_to_start_with_tampered_bundle(test_bundle: Path, tmp_path: Path) -> None:  # noqa: F811
    broken = tmp_path / "broken"
    shutil.copytree(test_bundle, broken)
    cal = json.loads((broken / "calibration.json").read_text())
    cal["a"] = 1.0  # alguien «ajusta» la calibración a mano
    (broken / "calibration.json").write_text(json.dumps(cal))
    with pytest.raises(BundleError, match="calibration.json"):
        with TestClient(build_app(broken, intra_op_threads=1)):
            pass
