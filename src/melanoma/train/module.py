"""LightningModule mínimo para clasificación binaria con logits."""

from __future__ import annotations

import lightning as L
import torch
from torch import nn


class LitBinaryClassifier(L.LightningModule):
    """Envuelve un `nn.Module` que devuelve logits `(batch, 1)`.

    Pérdida: BCE con logits. Optimizador: AdamW con `lr` y `weight_decay` de la config.
    """

    def __init__(self, model: nn.Module, lr: float, weight_decay: float) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x).squeeze(1)
        loss = self.criterion(logits, y.float())
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
