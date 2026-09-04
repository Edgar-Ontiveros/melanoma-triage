from __future__ import annotations

import torch
from omegaconf import DictConfig

from melanoma.models import build_model
from melanoma.utils import seed_everything


def test_seed_reproducible(smoke_cfg: DictConfig) -> None:
    seed = smoke_cfg.train.seed
    seed_everything(seed)
    a = torch.randn(4, 4)
    seed_everything(seed)
    b = torch.randn(4, 4)
    assert torch.equal(a, b)


def test_seed_reproducible_model_init(smoke_cfg: DictConfig) -> None:
    def build() -> torch.nn.Module:
        seed_everything(smoke_cfg.train.seed)
        model, _ = build_model(
            backbone=smoke_cfg.model.backbone,
            pretrained=False,
            num_classes=smoke_cfg.model.num_classes,
            dropout=smoke_cfg.model.dropout,
        )
        return model

    w1 = build().head.fc.weight
    w2 = build().head.fc.weight
    assert torch.equal(w1, w2)
