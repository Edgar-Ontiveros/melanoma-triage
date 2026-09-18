"""Módulo de entrenamiento (F2.2): detector de colapso con caso forzado, scheduler con
calentamiento y pos_weight en la pérdida."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from melanoma.models import build_model
from melanoma.train import (
    LitBinaryClassifier,
    cosine_with_warmup,
    cosine_with_warmup_factor,
    detect_collapse,
)


def test_collapse_detected_on_constant_probs(smoke_cfg: DictConfig) -> None:
    thr, tol = smoke_cfg.train.collapse_std_threshold, smoke_cfg.train.collapse_auc_tolerance
    report = detect_collapse(np.full(500, 0.017), auroc=0.5, std_threshold=thr, auc_tolerance=tol)
    assert report.collapsed
    assert "std" in report.reason and "AUROC" in report.reason


def test_collapse_detected_on_random_auc_only(smoke_cfg: DictConfig) -> None:
    thr, tol = smoke_cfg.train.collapse_std_threshold, smoke_cfg.train.collapse_auc_tolerance
    probs = np.random.default_rng(0).random(500)
    report = detect_collapse(probs, auroc=0.505, std_threshold=thr, auc_tolerance=tol)
    assert report.collapsed and "AUROC" in report.reason and "std" not in report.reason


def test_no_collapse_on_healthy_model(smoke_cfg: DictConfig) -> None:
    thr, tol = smoke_cfg.train.collapse_std_threshold, smoke_cfg.train.collapse_auc_tolerance
    probs = np.random.default_rng(0).random(500)
    assert not detect_collapse(probs, auroc=0.85, std_threshold=thr, auc_tolerance=tol).collapsed


class _ConstantModel(torch.nn.Module):
    """Devuelve siempre el mismo logit: un modelo colapsado a propósito."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor([-4.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias.expand(x.shape[0], 1)


def test_forced_collapse_is_logged(smoke_cfg: DictConfig, caplog: pytest.LogCaptureFixture) -> None:
    """Caso forzado (criterio 8): un modelo que predice una constante dispara la alerta."""
    import lightning as L
    from torch.utils.data import DataLoader, TensorDataset

    lit = LitBinaryClassifier(
        _ConstantModel(),
        lr=smoke_cfg.train.lr,
        weight_decay=0.0,
        collapse_std_threshold=smoke_cfg.train.collapse_std_threshold,
        collapse_auc_tolerance=smoke_cfg.train.collapse_auc_tolerance,
    )
    x = torch.randn(32, 3)
    y = (torch.arange(32) % 8 == 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    with caplog.at_level(logging.WARNING, logger="melanoma.train.module"):
        trainer.fit(lit, train_dataloaders=loader, val_dataloaders=loader)
    assert lit.collapse_history and lit.collapse_history[-1].collapsed
    assert any("COLAPSO DETECTADO" in r.message for r in caplog.records)
    assert trainer.callback_metrics["val/collapsed"].item() == 1.0


def test_pos_weight_scales_positive_loss(smoke_cfg: DictConfig) -> None:
    model, _ = build_model(
        backbone=smoke_cfg.model.backbone,
        pretrained=False,
        num_classes=smoke_cfg.model.num_classes,
        dropout=0.0,
    )
    plain = LitBinaryClassifier(model, lr=1e-3, weight_decay=0.0, pos_weight=None)
    heavy = LitBinaryClassifier(model, lr=1e-3, weight_decay=0.0, pos_weight=10.0)
    logits, y = torch.zeros(4), torch.ones(4)
    assert torch.isclose(heavy.criterion(logits, y), 10.0 * plain.criterion(logits, y))
    y0 = torch.zeros(4)
    assert torch.isclose(heavy.criterion(logits, y0), plain.criterion(logits, y0))


def test_cosine_with_warmup_shape() -> None:
    warmup, total = 10, 100
    f = [cosine_with_warmup_factor(s, warmup, total) for s in range(total + 1)]
    assert f[0] == pytest.approx(0.1) and f[warmup - 1] == pytest.approx(1.0)
    assert f[warmup] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(f[warmup:], f[warmup + 1 :], strict=False))
    assert f[total] == pytest.approx(0.0, abs=1e-9)
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = cosine_with_warmup(opt, warmup, total)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1)
    for _ in range(warmup):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0)


def test_freeze_backbone_trains_only_the_head(smoke_cfg: DictConfig) -> None:
    from melanoma.train.run import freeze_backbone

    model, _ = build_model(
        backbone=smoke_cfg.model.backbone,
        pretrained=False,
        num_classes=smoke_cfg.model.num_classes,
        dropout=0.0,
    )
    n_frozen = freeze_backbone(model)
    assert n_frozen > 0
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    lit = LitBinaryClassifier(model, lr=1e-3, weight_decay=0.0)
    opt = lit.configure_optimizers()
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_opt == sum(p.numel() for p in model.head.parameters())


def test_collapse_reports_nan_as_divergence(smoke_cfg: DictConfig) -> None:
    thr, tol = smoke_cfg.train.collapse_std_threshold, smoke_cfg.train.collapse_auc_tolerance
    probs = np.array([0.1, np.nan, 0.3, np.nan])
    report = detect_collapse(probs, auroc=float("nan"), std_threshold=thr, auc_tolerance=tol)
    assert report.collapsed and "divergió" in report.reason
