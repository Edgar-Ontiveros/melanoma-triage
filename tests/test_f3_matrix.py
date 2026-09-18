"""Comparación controlada de F3: lote efectivo 128, hiperparámetros fijos compartidos con B1 y
backbones que existen en timm. Sin descargas."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from tests.conftest import CONFIGS

MATRIX = ["b1_resnet50_224", "m1_effnetv2s_384", "m2_convnext_t_384", "a1_effnetv2s_224"]
EFFECTIVE_BATCH = 128
FIXED_TRAIN = [
    "lr",
    "weight_decay",
    "scheduler",
    "warmup_epochs",
    "max_epochs",
    "early_stopping_patience",
    "pos_weight",
    "precision",
    "monitor",
    "monitor_mode",
]
FIXED_DATA = ["augment", "val_resize", "sampler", "preprocess_scale"]
FIXED_MODEL = ["dropout", "freeze_backbone", "num_classes", "pretrained"]


def _plain(value):
    return OmegaConf.to_container(value) if OmegaConf.is_config(value) else value


def _compose(name: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        return compose(config_name="config", overrides=[f"+experiment={name}"])


@pytest.mark.parametrize("name", MATRIX)
def test_effective_batch_is_128(name: str) -> None:
    cfg = _compose(name)
    assert cfg.data.batch_size * cfg.train.accumulate_grad_batches == EFFECTIVE_BATCH, name


def test_matrix_configs_share_fixed_params() -> None:
    """Solo backbone, resolución y lote físico pueden diferir entre B1, M1, M2 y A1."""
    cfgs = {name: _compose(name) for name in MATRIX}
    ref = cfgs["b1_resnet50_224"]
    for name, cfg in cfgs.items():
        for key in FIXED_TRAIN:
            assert cfg.train[key] == ref.train[key], f"{name}: train.{key} difiere de B1"
        for key in FIXED_DATA:
            assert _plain(cfg.data[key]) == _plain(ref.data[key]), (
                f"{name}: data.{key} difiere de B1"
            )
        for key in FIXED_MODEL:
            assert cfg.model[key] == ref.model[key], f"{name}: model.{key} difiere de B1"
    assert {cfgs[n].model.backbone for n in MATRIX} == {
        "resnet50.a1_in1k",
        "tf_efficientnetv2_s.in21k_ft_in1k",
        "convnext_tiny.fb_in22k_ft_in1k_384",
    }
    assert cfgs["a1_effnetv2s_224"].data.image_size == 224
    assert cfgs["m1_effnetv2s_384"].data.image_size == 384
    assert cfgs["m2_convnext_t_384"].data.image_size == 384


@pytest.mark.parametrize("name", ["m1_effnetv2s_384", "m2_convnext_t_384", "a1_effnetv2s_224"])
def test_backbones_exist_in_timm(name: str) -> None:
    import timm

    backbone = _compose(name).model.backbone
    assert backbone in timm.list_pretrained(), f"{backbone} no tiene pesos preentrenados en timm"
    model = timm.create_model(backbone, pretrained=False, num_classes=0)
    assert model.num_features > 0
