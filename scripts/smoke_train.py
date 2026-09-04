"""Prueba de humo de entrenamiento.

Recorre el camino completo config → semilla → fábrica → LightningModule → Trainer
con datos sintéticos en memoria. No descarga pesos ni datasets. Es el gate que se
corre antes de encender cualquier instancia de GPU pagada.

Uso: ``python scripts/smoke_train.py`` (configuración en ``configs/smoke.yaml``).
"""

from __future__ import annotations

import time

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from melanoma.models import build_model
from melanoma.train import LitBinaryClassifier
from melanoma.utils import seed_everything


def make_synthetic_loader(cfg: DictConfig, data_config: dict) -> DataLoader:
    """Ruido gaussiano con la forma que espera el backbone y etiquetas binarias aleatorias."""
    n = cfg.data.synthetic_samples
    channels = data_config["input_size"][0]
    size = cfg.data.image_size
    x = torch.randn(n, channels, size, size)
    y = torch.randint(0, cfg.model.num_classes + 1, (n,))
    return DataLoader(
        TensorDataset(x, y),
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )


@hydra.main(config_path="../configs", config_name="smoke", version_base="1.3")
def main(cfg: DictConfig) -> None:
    start = time.perf_counter()
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.train.seed, deterministic=cfg.train.deterministic)

    model, data_config = build_model(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    )
    print(f"data_config del backbone: {data_config}")

    loader = make_synthetic_loader(cfg, data_config)
    lit = LitBinaryClassifier(model, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    trainer = L.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        precision=cfg.train.precision,
        deterministic=cfg.train.deterministic,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(lit, train_dataloaders=loader)

    elapsed = time.perf_counter() - start
    print(f"SMOKE OK: {cfg.train.max_epochs} épocas, {elapsed:.1f} s")


if __name__ == "__main__":
    main()
