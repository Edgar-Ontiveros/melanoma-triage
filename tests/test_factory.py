from __future__ import annotations

import pytest
import torch
from omegaconf import DictConfig

from melanoma.models import build_model

# Dos backbones distintos, sin pesos: no descarga nada.
BACKBONES = ["resnet18", "tf_efficientnetv2_s"]


@pytest.mark.parametrize("backbone", BACKBONES)
def test_factory_output_shape(backbone: str, smoke_cfg: DictConfig) -> None:
    model, data_config = build_model(
        backbone=backbone,
        pretrained=False,
        num_classes=smoke_cfg.model.num_classes,
        dropout=smoke_cfg.model.dropout,
    )
    model.eval()
    batch = smoke_cfg.data.batch_size
    x = torch.randn(batch, *data_config["input_size"])
    with torch.no_grad():
        out = model(x)
    assert out.shape == (batch, smoke_cfg.model.num_classes)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_data_config_from_timm(backbone: str, smoke_cfg: DictConfig) -> None:
    _, data_config = build_model(
        backbone=backbone,
        pretrained=False,
        num_classes=smoke_cfg.model.num_classes,
        dropout=smoke_cfg.model.dropout,
    )
    for key in ("input_size", "mean", "std", "interpolation"):
        assert key in data_config, f"falta {key} en data_config"
    mean, std = tuple(data_config["mean"]), tuple(data_config["std"])
    channels = data_config["input_size"][0]
    assert len(mean) == channels and len(std) == channels
    assert mean != tuple([0.0] * channels), "mean trivial: no viene del checkpoint"
    assert std != tuple([1.0] * channels), "std trivial: no viene del checkpoint"
    assert all(0.0 < m < 1.0 for m in mean)
    assert all(0.0 < s < 1.0 for s in std)
