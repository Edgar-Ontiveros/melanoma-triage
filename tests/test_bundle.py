"""F6: paquete de modelo (exportación, manifiesto, preprocesamiento desde JSON, paridad ONNX
con el paquete de prueba, CAM de la API ≡ CAM de melanoma.explain). Todo con pesos aleatorios;
lo que necesita el checkpoint real está en ``@pytest.mark.local``."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from services.api.bundle import BundleError, load_bundle, verify_manifest
from services.api.cam import cam_from_features as api_cam
from services.api.cam import upsample as api_upsample
from services.api.inference import Predictor
from services.api.preprocess import Preprocessor, resize_smallest_side

from melanoma.data.transforms import build_transforms
from melanoma.explain import cam_from_features as ref_cam
from melanoma.explain import upsample as ref_upsample
from melanoma.models import build_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def test_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Paquete de prueba (pesos aleatorios, semilla 0) construido una vez por sesión."""
    from scripts.make_test_bundle import build

    out = tmp_path_factory.mktemp("bundle")
    build(out, seed=0)
    return out


@pytest.fixture(scope="session")
def negative_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Pesos de la capa lineal en cero y sesgo −50: logit muy negativo y CAM nulo (con pesos
    negativos no basta: las activaciones tras SiLU pueden ser negativas y ReLU(w·A) > 0)."""
    from scripts.make_test_bundle import build

    out = tmp_path_factory.mktemp("bundle_neg")
    build(out, seed=0, negative_bias=-50.0, zero_weights=True)
    return out


def synthetic_rgb(h: int = 512, w: int = 768, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (210, 165, 150), dtype=np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    disc = (ys - h / 2) ** 2 + (xs - w / 2) ** 2 < (0.2 * min(h, w)) ** 2
    img[disc] = (95, 55, 45)
    img += rng.normal(0, 6, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- manifiesto


def test_bundle_has_seven_files(test_bundle: Path) -> None:
    names = {p.name for p in test_bundle.iterdir()}
    assert names == {
        "model.onnx",
        "preprocess.json",
        "calibration.json",
        "thresholds.json",
        "cam_weights.npy",
        "metrics.json",
        "manifest.json",
    }
    manifest = json.loads((test_bundle / "manifest.json").read_text())
    assert manifest["onnx"]["opset"] == {"ai.onnx": 17}
    assert manifest["onnx"]["outputs"] == ["logit", "features"]
    assert manifest["onnx"]["features_shape"] == [1280, 7, 7]
    assert manifest["kind"] == "test"


def test_bundle_manifest_sha256(test_bundle: Path, tmp_path: Path) -> None:
    """Cada archivo coincide con su SHA; un byte cambiado → no arranca."""
    verify_manifest(test_bundle)
    import shutil

    broken = tmp_path / "broken"
    shutil.copytree(test_bundle, broken)
    path = broken / "thresholds.json"
    data = bytearray(path.read_bytes())
    data[-2] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(BundleError, match="SHA256"):
        load_bundle(broken)
    incomplete = tmp_path / "incomplete"
    shutil.copytree(test_bundle, incomplete)
    (incomplete / "cam_weights.npy").unlink()
    with pytest.raises(BundleError, match="falta"):
        load_bundle(incomplete)


def test_bundle_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_BUNDLE_DIR", raising=False)
    with pytest.raises(BundleError, match="MODEL_BUNDLE_DIR"):
        load_bundle()


# ---------------------------------------------------------------- preprocesamiento


def test_preprocess_from_json_only(test_bundle: Path) -> None:
    """Cambiar ``mean`` en preprocess.json cambia la salida; no hay constantes en services/api."""
    spec = json.loads((test_bundle / "preprocess.json").read_text())
    img = synthetic_rgb()
    crop, x = Preprocessor(spec)(img)
    assert crop.shape == (224, 224, 3) and crop.dtype == np.uint8
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    altered = dict(spec, mean=[m + 0.1 for m in spec["mean"]])
    _, x2 = Preprocessor(altered)(img)
    assert not np.allclose(x, x2)
    expected = np.broadcast_to(0.1 / np.asarray(spec["std"])[None, :, None, None], x.shape)
    np.testing.assert_allclose(x - x2, expected, atol=1e-5)

    forbidden = re.compile(r"0\.485|0\.456|0\.406|0\.229|0\.224|0\.225|(?<![\d.])0\.\d{3}(?![\d])")
    offenders = []
    for path in sorted((ROOT / "services" / "api").rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if forbidden.search(code):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, "constantes de preprocesamiento en services/api:\n" + "\n".join(offenders)


def test_preprocess_matches_training_transform(test_bundle: Path) -> None:
    """El recorte de la API reproduce SmallestMaxSize + CenterCrop de albumentations (cv2
    INTER_CUBIC) hasta el redondeo de unos pocos píxeles, y el tensor coincide con Normalize."""
    spec = json.loads((test_bundle / "preprocess.json").read_text())
    _, data_config = build_model(
        "tf_efficientnetv2_s.in21k_ft_in1k", pretrained=False, num_classes=1, dropout=0.2
    )
    tf = build_transforms(data_config, 224, None, train=False, val_resize="center_crop")
    for seed, (h, w) in enumerate(((512, 768), (768, 512), (300, 300), (1024, 1500))):
        img = synthetic_rgb(h, w, seed)
        crop, x = Preprocessor(spec)(img)
        ref = tf(image=img)["image"].numpy()[None]
        assert np.abs(x - ref).max() < 0.02, (h, w)  # ±1 píxel / 255 / 0.5 ≈ 0.008
        assert (np.abs(x - ref) > 1e-6).mean() < 0.001, (h, w)


def test_resize_keeps_aspect_and_rounds_like_albumentations() -> None:
    img = np.zeros((300, 450, 3), dtype=np.uint8)
    out = resize_smallest_side(img, 224)
    assert out.shape == (224, 336, 3)
    assert resize_smallest_side(np.zeros((224, 224, 3), dtype=np.uint8), 224).shape == (224, 224, 3)


# ---------------------------------------------------------------- paridad y CAM


def test_onnx_parity_synthetic(test_bundle: Path) -> None:
    """Paquete de prueba: ONNX vs torch con los mismos pesos, ``|Δ| < 1e-4`` relativo.

    Con pesos aleatorios (BatchNorm sin estadísticas aprendidas) los logits llegan a
    cientos y las activaciones a miles; la tolerancia absoluta de 1e-4 de la spec está por
    debajo de la resolución de float32 a esa escala, así que se aplica relativa a
    ``max(1, |valor|)``. Con el checkpoint real (F6.2) la diferencia absoluta es 2e-5.
    """
    from scripts.make_test_bundle import BACKBONE

    torch.manual_seed(0)
    model, _ = build_model(BACKBONE, pretrained=False, num_classes=1, dropout=0.2)
    model.eval()
    bundle = load_bundle(test_bundle)
    predictor = Predictor(bundle, intra_op_threads=2)
    pre = Preprocessor(bundle.preprocess)
    x = torch.tensor(np.concatenate([pre(synthetic_rgb(seed=s))[1] for s in range(3)]))
    with torch.no_grad():
        feats = model.backbone.forward_features(x)
        logit = model.head(feats.mean(dim=(2, 3)))
    o_logit, o_feats = predictor.session.run(
        predictor.output_names, {predictor.input_name: x.numpy()}
    )
    rel_logit = np.abs(o_logit - logit.numpy()) / np.maximum(1.0, np.abs(logit.numpy()))
    rel_feats = np.abs(o_feats - feats.numpy()) / np.maximum(1.0, np.abs(feats.numpy()))
    assert rel_logit.max() < 1e-4, rel_logit.max()
    assert rel_feats.max() < 1e-3, rel_feats.max()
    np.testing.assert_allclose(
        bundle.cam_weights, model.head.fc.weight.detach().numpy().reshape(-1)
    )


def test_api_cam_equals_explain_cam() -> None:
    rng = np.random.default_rng(4)
    feats = rng.normal(size=(1280, 7, 7)).astype(np.float32)
    w = rng.normal(size=1280).astype(np.float32)
    np.testing.assert_allclose(api_cam(feats, w), ref_cam(feats, w), atol=1e-6)
    cam = api_cam(feats, w)
    np.testing.assert_allclose(api_upsample(cam, 224), ref_upsample(cam, 224), atol=1e-6)
    zero = api_cam(np.ones((4, 7, 7), np.float32), -np.ones(4, np.float32))
    assert not zero.any()


def test_predictor_no_positive_evidence(negative_bundle: Path) -> None:
    pred = Predictor(load_bundle(negative_bundle), 2).predict_array(synthetic_rgb())
    assert pred.cam_status == "no_positive_evidence"
    assert not pred.cam_grid.any()
    assert pred.probability < 1e-6 and not pred.refer


def test_predictor_rejects_bad_bytes(test_bundle: Path, tmp_path: Path) -> None:
    from services.api.inference import InputError

    predictor = Predictor(load_bundle(test_bundle), 2)
    with pytest.raises(InputError) as e:
        predictor.predict_bytes(b"no soy una imagen")
    assert e.value.status == 422
    small = tmp_path / "small.png"
    Image.fromarray(synthetic_rgb(32, 32)).save(small)
    with pytest.raises(InputError) as e:
        predictor.predict_bytes(small.read_bytes())
    assert e.value.status == 422
    gif = tmp_path / "x.gif"
    Image.fromarray(synthetic_rgb(128, 128)).save(gif)
    with pytest.raises(InputError) as e:
        predictor.predict_bytes(gif.read_bytes())
    assert e.value.status == 415


# ---------------------------------------------------------------- checkpoint real


@pytest.mark.local
def test_real_bundle_matches_checkpoint_and_parity() -> None:
    """Con el checkpoint real: el paquete existe, su manifiesto es válido y la paridad de
    F6.2 pasó en las cuatro comparaciones de tensores con decisiones idénticas."""
    bundle_dir = ROOT / "models" / "bundle"
    if not bundle_dir.exists():
        pytest.skip("models/bundle no existe (make export-bundle)")
    bundle = load_bundle(bundle_dir)
    expected = json.loads((ROOT / "reports" / "f3_final_model.json").read_text())["sha256"]
    assert bundle.manifest["checkpoint"]["sha256"] == expected
    parity = json.loads((ROOT / "reports" / "f6_parity.json").read_text())
    assert parity["model_version"] == bundle.model_version
    for key in ("logit", "platt", "cam", "decision_tau95"):
        assert parity[key]["pass"], key
    assert parity["full_chain"]["decisions_identical"]
