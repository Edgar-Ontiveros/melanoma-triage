"""DataModule (F2.1): carga train/val, bloquea test, normaliza desde el data_config y solo
aumenta entrenamiento. Usa un dataset sintético en disco; sin descargas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from melanoma.data import (
    ISICDataModule,
    LockedTestSplitError,
    datamodule_from_config,
    detect_env,
    resolve_paths,
    verify_split_hashes,
)
from melanoma.data.synthetic import write_synthetic_dataset
from melanoma.data.transforms import build_transforms
from melanoma.models import build_model

N_IMAGES = 40


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("synthetic")
    return write_synthetic_dataset(root, n_images=N_IMAGES, positive_rate=0.25, seed=0)


@pytest.fixture(scope="module")
def data_config(smoke_cfg_module: DictConfig) -> dict:
    _, dc = build_model(
        backbone=smoke_cfg_module.model.backbone,
        pretrained=False,
        num_classes=smoke_cfg_module.model.num_classes,
        dropout=smoke_cfg_module.model.dropout,
    )
    return dc


@pytest.fixture(scope="module")
def smoke_cfg_module() -> DictConfig:
    from hydra import compose, initialize_config_dir

    from tests.conftest import CONFIGS

    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        return compose(config_name="smoke")


def _dm(synthetic: dict, data_config: dict, cfg: DictConfig, **kw) -> ISICDataModule:
    args = dict(
        manifest_path=synthetic["manifest_path"],
        splits_dir=synthetic["splits_dir"],
        images_dir=synthetic["images_dir"],
        data_config=data_config,
        batch_size=8,
        num_workers=0,
        image_size=cfg.data.image_size,
        augment=OmegaConf.to_container(cfg.data.augment),
    )
    args.update(kw)
    dm = ISICDataModule(**args)
    dm.setup("fit")
    return dm


def test_loads_train_and_val(synthetic, data_config, smoke_cfg_module) -> None:
    dm = _dm(synthetic, data_config, smoke_cfg_module)
    assert len(dm.train_ds) + len(dm.val_ds) == N_IMAGES
    x, y = next(iter(dm.train_dataloader()))
    size = smoke_cfg_module.data.image_size
    assert x.shape == (8, data_config["input_size"][0], size, size)
    assert x.dtype == torch.float32 and y.dtype == torch.long
    xv, _ = next(iter(dm.val_dataloader()))
    assert xv.shape[1:] == x.shape[1:]


def test_test_dataloader_raises(synthetic, data_config, smoke_cfg_module) -> None:
    dm = _dm(synthetic, data_config, smoke_cfg_module)
    with pytest.raises(LockedTestSplitError):
        dm.test_dataloader()
    with pytest.raises(LockedTestSplitError):
        dm.setup("test")


def test_val_has_no_augmentation(synthetic, data_config, smoke_cfg_module) -> None:
    dm = _dm(synthetic, data_config, smoke_cfg_module)
    a = torch.stack([dm.val_ds[i][0] for i in range(4)])
    b = torch.stack([dm.val_ds[i][0] for i in range(4)])
    assert torch.equal(a, b), "validación debe ser determinista (sin aumentación)"


def test_train_is_augmented(synthetic, data_config, smoke_cfg_module) -> None:
    dm = _dm(synthetic, data_config, smoke_cfg_module)
    draws = torch.stack([dm.train_ds[0][0] for _ in range(6)])
    assert not all(torch.equal(draws[0], d) for d in draws[1:]), "train debe variar entre lecturas"


def test_normalization_comes_from_data_config(data_config, smoke_cfg_module) -> None:
    """Un píxel constante v debe salir como (v/255 - mean)/std con los valores del backbone."""
    tf = build_transforms(data_config, smoke_cfg_module.data.image_size, None, train=False)
    v = 200
    img = np.full((30, 40, 3), v, dtype=np.uint8)
    out = tf(image=img)["image"][:, 0, 0].numpy()
    expected = (v / 255.0 - np.array(data_config["mean"])) / np.array(data_config["std"])
    assert np.allclose(out, expected, atol=1e-4)


def test_preprocess_scale_bug_conditions(data_config, smoke_cfg_module) -> None:
    """B comprime la señal ~255x (dos píxeles distintos salen casi iguales); C la infla ~255x."""
    size = smoke_cfg_module.data.image_size
    dark = np.full((30, 40, 3), 50, dtype=np.uint8)
    light = np.full((30, 40, 3), 200, dtype=np.uint8)

    def contrast(scale: str) -> float:
        tf = build_transforms(data_config, size, None, False, scale)
        return (tf(image=light)["image"] - tf(image=dark)["image"]).abs().max().item()

    ok, small, big = contrast("none"), contrast("divide255"), contrast("multiply255")
    assert small == pytest.approx(ok / 255.0, rel=1e-3)
    assert big == pytest.approx(ok * 255.0, rel=1e-3)
    with pytest.raises(ValueError):
        build_transforms(data_config, size, None, False, "otra")


def _circle_image(width: int = 600, height: int = 400, radius: int = 120) -> np.ndarray:
    """Imagen 3:2 negra con un círculo blanco centrado."""
    yy, xx = np.mgrid[:height, :width]
    mask = (xx - width / 2) ** 2 + (yy - height / 2) ** 2 <= radius**2
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[mask] = 255
    return img


def _blob_aspect(out: torch.Tensor) -> float:
    """Ancho/alto de la caja del blob más brillante en el canal 0 (tras normalizar)."""
    ch = out[0].numpy()
    mask = ch > (ch.min() + ch.max()) / 2
    ys, xs = np.nonzero(mask)
    return (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)


def test_val_center_crop_preserves_aspect_ratio(data_config, smoke_cfg_module) -> None:
    """Un círculo en una imagen 3:2 sigue siendo circular tras la transformación de validación."""
    size = smoke_cfg_module.data.image_size
    img = _circle_image()
    kept = build_transforms(data_config, size, None, train=False, val_resize="center_crop")
    squashed = build_transforms(data_config, size, None, train=False, val_resize="squash")
    out = kept(image=img)["image"]
    assert out.shape[1:] == (size, size)
    assert _blob_aspect(out) == pytest.approx(1.0, abs=0.05)
    # Control: el aplastado comprime el ancho 1.5x más que el alto → el círculo sale con
    # proporción 2/3 (un óvalo vertical).
    assert _blob_aspect(squashed(image=img)["image"]) == pytest.approx(2 / 3, abs=0.05)
    assert smoke_cfg_module.data.val_resize == "center_crop"
    with pytest.raises(ValueError):
        build_transforms(data_config, size, None, train=False, val_resize="otra")


def test_weighted_sampler_raises_positive_rate(synthetic, data_config, smoke_cfg_module) -> None:
    torch.manual_seed(0)
    plain = _dm(synthetic, data_config, smoke_cfg_module, sampler="none")
    weighted = _dm(synthetic, data_config, smoke_cfg_module, sampler="weighted")
    neg, pos = plain.class_counts()
    assert pytest.approx(plain.pos_weight()) == neg / pos
    ys = torch.cat([y for _, y in weighted.train_dataloader()])
    rate = ys.float().mean().item()
    base = pos / (neg + pos)
    assert len(ys) == len(weighted.train_ds)
    assert rate > base, f"el muestreo ponderado debe subir la tasa de positivos ({rate} vs {base})"
    with pytest.raises(ValueError):
        _dm(synthetic, data_config, smoke_cfg_module, sampler="otro")


def test_val_frame_aligned_with_loader(synthetic, data_config, smoke_cfg_module) -> None:
    dm = _dm(synthetic, data_config, smoke_cfg_module)
    ys = torch.cat([y for _, y in dm.val_dataloader()]).numpy()
    frame = dm.val_frame()
    assert list(frame.columns) == ["image_id", "target", "patient_id"]
    assert np.array_equal(frame["target"].to_numpy(), ys)


def test_split_hashes_verified(synthetic) -> None:
    hashes = verify_split_hashes(synthetic["splits_dir"])
    assert set(hashes) == {"train", "val"}
    (synthetic["splits_dir"] / "SHA256SUMS").write_text("deadbeef  train.txt\n")
    with pytest.raises(RuntimeError):
        verify_split_hashes(synthetic["splits_dir"])


def test_resolve_paths_by_env(root_dir: Path, cfg: DictConfig) -> None:
    assert detect_env(environ={}, kaggle_root="/definitivamente/no/existe") == "local"
    assert detect_env(environ={"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}) == "kaggle"
    local = resolve_paths(OmegaConf.merge(cfg.data, {"env": "local"}), root=root_dir)
    kaggle = resolve_paths(OmegaConf.merge(cfg.data, {"env": "kaggle"}), root=root_dir)
    assert local.env == "local" and local.images_dir == root_dir / "data/processed/isic2020_512"
    assert kaggle.env == "kaggle" and str(kaggle.images_dir).startswith("/kaggle/input/")
    with pytest.raises(ValueError):
        resolve_paths(OmegaConf.merge(cfg.data, {"env": "nube"}), root=root_dir)


def test_datamodule_from_config(synthetic, data_config, smoke_cfg_module) -> None:
    data_cfg = OmegaConf.merge(
        smoke_cfg_module.data,
        {"env": "local", "paths": {"local": {k: str(v) for k, v in synthetic.items()}}},
    )
    dm = datamodule_from_config(data_cfg, data_config, root=Path("/"))
    dm.setup("fit")
    assert len(dm.val_ds) > 0
