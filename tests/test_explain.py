"""F5: contrato de CAM, equivalencia con Grad-CAM, prueba de cordura, máscaras, artefactos y
el guardarraíl de DDI."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from melanoma.explain import cam_from_features, energy_fraction, explain, upsample
from melanoma.explain.features import (
    extract_features,
    gradcam_reference,
    linear_weight,
    target_layer,
)
from melanoma.explain.masks import lesion_mask, mask_centroid_radius
from melanoma.explain.perturb import PERTURBATIONS, pixel_change
from melanoma.explain.sanity import SCOPES, randomized_copy, spearman_maps
from melanoma.models import build_model

ROOT = Path(__file__).resolve().parents[1]
BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m, _ = build_model(BACKBONE, pretrained=False, num_classes=1, dropout=0.2)
    return m.eval()


def synthetic_batch(n: int, size: int = 224, seed: int = 0) -> torch.Tensor:
    """Imágenes con estructura (un disco oscuro sobre fondo claro más ruido), normalizadas
    con media 0.5 / desviación 0.5 como el backbone."""
    rng = np.random.default_rng(seed)
    imgs = []
    for _ in range(n):
        img = np.full((size, size, 3), 200, dtype=np.float32)
        cy, cx = rng.uniform(0.3, 0.7, 2) * size
        r = rng.uniform(0.12, 0.25) * size
        ys, xs = np.mgrid[0:size, 0:size]
        disc = (ys - cy) ** 2 + (xs - cx) ** 2 < r**2
        img[disc] = rng.uniform(40, 90, 3)
        img += rng.normal(0, 8, img.shape)
        imgs.append(np.clip(img, 0, 255) / 255.0)
    x = torch.tensor(np.stack(imgs), dtype=torch.float32).permute(0, 3, 1, 2)
    return (x - 0.5) / 0.5


# ---------------------------------------------------------------- contrato


def test_cam_output_contract() -> None:
    rng = np.random.default_rng(1)
    feats = rng.normal(size=(1280, 7, 7)).astype(np.float32)
    w = rng.normal(size=1280).astype(np.float32)
    cam = cam_from_features(feats, w)
    assert cam.shape == (7, 7) and cam.dtype == np.float32
    assert cam.min() >= 0.0 and cam.max() == pytest.approx(1.0)
    assert explain is cam_from_features
    batch = cam_from_features(np.stack([feats, feats * 2]), w)
    assert batch.shape == (2, 7, 7)
    np.testing.assert_allclose(batch[0], batch[1], atol=1e-6)  # invariante a la escala
    for size in (224, 512, 7):
        up = upsample(cam, size)
        assert up.shape == (size, size) and up.dtype == np.float32
        assert up.min() >= -1e-6 and up.max() <= 1.0 + 1e-6


def test_cam_zero_when_no_positive_evidence() -> None:
    feats = np.ones((4, 7, 7), dtype=np.float32)
    cam = cam_from_features(feats, -np.ones(4, dtype=np.float32))
    assert cam.shape == (7, 7) and not cam.any()
    assert np.isnan(energy_fraction(cam, np.ones((7, 7), dtype=bool)))


def test_cam_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError):
        cam_from_features(np.zeros((3, 7, 7)), np.zeros(4))
    with pytest.raises(ValueError):
        cam_from_features(np.zeros((7, 7)), np.zeros(7))
    with pytest.raises(ValueError):
        upsample(np.zeros((1, 7, 7)), 10)


def test_upsample_matches_torch_bilinear() -> None:
    rng = np.random.default_rng(3)
    cam = rng.uniform(size=(7, 7)).astype(np.float32)
    ours = upsample(cam, 224)
    ref = F.interpolate(
        torch.tensor(cam)[None, None], size=(224, 224), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-5)


def test_energy_fraction() -> None:
    cam = np.zeros((4, 4))
    cam[:2, :2] = 1.0
    region = np.zeros((4, 4), dtype=bool)
    region[:2, :] = True
    assert energy_fraction(cam, region) == pytest.approx(1.0)
    region[:, :] = False
    region[2:, :] = True
    assert energy_fraction(cam, region) == pytest.approx(0.0)


# ---------------------------------------------------------------- equivalencia


def test_cam_equals_gradcam(model) -> None:
    """Sobre imágenes sintéticas, la CAM en numpy coincide con pytorch-grad-cam sobre
    ``backbone.bn2`` (correlación > 0.99 por imagen tras interpolar a 224 px)."""
    x = synthetic_batch(4)
    feats, logits = extract_features(model, x)
    assert feats.shape == (4, 1280, 7, 7)
    with torch.no_grad():
        np.testing.assert_allclose(logits, model(x).squeeze(1).numpy(), atol=1e-4)
    cams = cam_from_features(feats, linear_weight(model))
    ref = gradcam_reference(model, x)
    assert ref.shape == (4, 224, 224)
    assert target_layer(model) is model.backbone.bn2
    for i in range(4):
        ours = upsample(cams[i], 224)
        if ref[i].max() == 0 and ours.max() == 0:
            continue
        r = np.corrcoef(ours.ravel(), ref[i].ravel())[0, 1]
        assert r > 0.99, f"imagen {i}: correlación {r:.4f}"


# ---------------------------------------------------------------- cordura


def test_cam_depends_on_weights(model) -> None:
    """Reinicializar los pesos cambia el mapa: la CAM no es una función solo de la imagen."""
    x = synthetic_batch(3, seed=7)
    feats, _ = extract_features(model, x)
    cams = cam_from_features(feats, linear_weight(model))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rhos = []
    for scope in SCOPES:
        rand = randomized_copy(model, scope, seed=11)
        rf, _ = extract_features(rand, x)
        rcams = cam_from_features(rf, linear_weight(rand))
        assert not np.allclose(cams, rcams, atol=1e-3), scope
        rhos += [spearman_maps(cams[i], rcams[i]) for i in range(3)]
    rhos = np.array([r for r in rhos if not np.isnan(r)])
    assert rhos.size and rhos.mean() < 0.5, rhos
    for k, v in model.state_dict().items():  # el original no se toca
        assert torch.equal(v, before[k]), k


def test_randomized_copy_rejects_unknown_scope(model) -> None:
    with pytest.raises(ValueError):
        randomized_copy(model, "nope", seed=0)


def test_spearman_constant_map_is_nan() -> None:
    assert np.isnan(spearman_maps(np.zeros(9), np.arange(9)))
    assert spearman_maps(np.arange(9), np.arange(9)) == pytest.approx(1.0)


# ---------------------------------------------------------------- máscaras


def synthetic_rgb(size: int = 224, radius: float = 40.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), (215, 170, 150), dtype=np.float32)
    ys, xs = np.mgrid[0:size, 0:size]
    disc = (ys - size / 2) ** 2 + (xs - size / 2) ** 2 < radius**2
    img[disc] = (90, 50, 40)
    img += rng.normal(0, 4, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def test_lesion_mask_on_synthetic_disc() -> None:
    img = synthetic_rgb(radius=40)
    mask, info = lesion_mask(img)
    assert info.valid and info.reason == "ok"
    expected = np.pi * 40**2 / 224**2
    assert info.area_frac == pytest.approx(expected, rel=0.15)
    cy, cx, r = mask_centroid_radius(mask)
    assert abs(cy - 112) < 3 and abs(cx - 112) < 3 and abs(r - 40) < 4


def test_lesion_mask_filters() -> None:
    _, tiny = lesion_mask(synthetic_rgb(radius=8))  # < 3 % del área
    assert not tiny.valid and tiny.reason == "too_small"
    _, huge = lesion_mask(synthetic_rgb(radius=108))  # toca el borde o > 70 %
    assert not huge.valid
    flat = np.full((64, 64, 3), 180, dtype=np.uint8)
    mask, info = lesion_mask(flat)
    assert not info.valid and not mask.any()


# ---------------------------------------------------------------- artefactos


@pytest.mark.parametrize("name", sorted(PERTURBATIONS))
def test_perturbations_preserve_shape(name: str) -> None:
    img = synthetic_rgb()
    mask, _ = lesion_mask(img)
    rng = np.random.default_rng(5)
    out, region = PERTURBATIONS[name](img, rng, mask)
    assert out.shape == img.shape and out.dtype == img.dtype
    assert region.shape == img.shape[:2] and region.dtype == bool and region.any()
    assert pixel_change(img, out) > 0.5
    if name == "ink":  # la tinta no cubre la lesión
        assert not (region & mask).any()
    changed = np.abs(out.astype(int) - img.astype(int)).sum(axis=2) > 0
    if name == "vignette":  # el centro queda intacto y las esquinas se oscurecen
        assert not changed[100:124, 100:124].any()
        assert (out[0, 0].astype(int) < img[0, 0].astype(int)).all()
    elif name != "noise":  # el cambio ocurre dentro de la región declarada
        assert (changed & ~region).mean() < 0.02


def test_perturbations_are_reproducible() -> None:
    img = synthetic_rgb()
    for name, fn in PERTURBATIONS.items():
        a, _ = fn(img, np.random.default_rng(9), None)
        b, _ = fn(img, np.random.default_rng(9), None)
        assert np.array_equal(a, b), name


# ---------------------------------------------------------------- DDI


def test_no_ddi_in_explain() -> None:
    """Ningún módulo ni script de F5 referencia rutas de DDI (el acuerdo prohíbe usar sus
    imágenes en mapas o figuras). Se vigilan rutas y archivos, no la palabra: el reporte
    declara en prosa que no se usa DDI."""
    ddi = re.compile(r"raw/ddi|ddi_metadata|ddi/images|ddi\.root|ddi_results", re.I)
    files = sorted((ROOT / "src/melanoma/explain").glob("*.py")) + sorted(
        (ROOT / "scripts").glob("f5_*.py")
    )
    assert files
    offenders = [p.name for p in files if ddi.search(p.read_text(encoding="utf-8"))]
    assert not offenders, offenders
