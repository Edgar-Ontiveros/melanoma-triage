"""LightningModule para clasificación binaria con logits (F0, extendido en F2.2).

Qué agrega F2.2 sobre el mínimo de F0: ``pos_weight`` en la pérdida, AUROC y AUPRC de
validación con torchmetrics, cosine annealing con calentamiento, detección de colapso al
final de cada validación y ``predict_step`` que devuelve probabilidades.
"""

from __future__ import annotations

import logging
from typing import Any

import lightning as L
import numpy as np
import torch
from torch import nn
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from melanoma.train.collapse import CollapseReport, detect_collapse
from melanoma.train.schedulers import cosine_with_warmup

log = logging.getLogger(__name__)


class LitBinaryClassifier(L.LightningModule):
    """Envuelve un `nn.Module` que devuelve logits `(batch, 1)`.

    Pérdida: BCE con logits y ``pos_weight``. Optimizador: AdamW con `lr` y `weight_decay`.
    Scheduler: cosine annealing por paso con ``warmup_epochs`` de calentamiento (o ninguno).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        pos_weight: float | None = None,
        scheduler: str = "none",
        warmup_epochs: float = 0.0,
        collapse_std_threshold: float | None = None,
        collapse_auc_tolerance: float | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.pos_weight = 1.0 if pos_weight is None else float(pos_weight)
        self.scheduler_name = scheduler
        self.warmup_epochs = float(warmup_epochs)
        self.collapse_std_threshold = collapse_std_threshold
        self.collapse_auc_tolerance = collapse_auc_tolerance
        self.save_hyperparameters(ignore=["model"])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]))
        self.val_auroc = BinaryAUROC()
        self.val_auprc = BinaryAveragePrecision()
        self._val_probs: list[torch.Tensor] = []
        self.collapse_history: list[CollapseReport] = []
        # Una fila por época de validación (curvas de F2.6 sin depender de W&B).
        self.epoch_history: list[dict[str, float | int | bool]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ---- entrenamiento -------------------------------------------------------------------
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x).squeeze(1)
        loss = self.criterion(logits, y.float())
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # ---- validación ----------------------------------------------------------------------
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        logits = self(x).squeeze(1)
        loss = self.criterion(logits, y.float())
        probs = torch.sigmoid(logits.float())
        self.val_auroc.update(probs, y.int())
        self.val_auprc.update(probs, y.int())
        self._val_probs.append(probs.detach().cpu())
        self.log("val/loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_probs = []

    def on_validation_epoch_end(self) -> None:
        auroc = self.val_auroc.compute().item()
        auprc = self.val_auprc.compute().item()
        self.log("val/auroc", auroc, prog_bar=True)
        self.log("val/auprc", auprc, prog_bar=True)
        self.val_auroc.reset()
        self.val_auprc.reset()
        probs = torch.cat(self._val_probs).numpy() if self._val_probs else np.zeros(0)
        self._val_probs = []
        if self.collapse_std_threshold is None or self.collapse_auc_tolerance is None:
            return
        if self.trainer.sanity_checking:  # modelo sin entrenar: un "colapso" aquí no informa nada
            return
        report = detect_collapse(
            probs, auroc, self.collapse_std_threshold, self.collapse_auc_tolerance
        )
        self.collapse_history.append(report)
        val_loss = self.trainer.callback_metrics.get("val/loss")
        self.epoch_history.append(
            {
                "epoch": int(self.current_epoch),
                "val/loss": float(val_loss) if val_loss is not None else float("nan"),
                "val/auroc": auroc,
                "val/auprc": auprc,
                "val/prob_std": report.prob_std,
                "val/prob_mean": report.prob_mean,
                "collapsed": report.collapsed,
            }
        )
        self.log("val/prob_std", report.prob_std)
        self.log("val/collapsed", float(report.collapsed))
        if report.collapsed:
            msg = (
                f"COLAPSO DETECTADO en la época {self.current_epoch}: {report.reason} "
                f"(media de probabilidades {report.prob_mean:.4f}). El modelo predice una sola "
                "clase; revisar preprocesamiento, pos_weight/sampler y tasa de aprendizaje."
            )
            log.warning(msg)
            print(f"\n*** {msg}\n", flush=True)

    def on_train_epoch_end(self) -> None:
        loss = self.trainer.callback_metrics.get("train/loss_epoch")
        if loss is not None and self.epoch_history:
            last = self.epoch_history[-1]
            if last["epoch"] == self.current_epoch:
                last["train/loss"] = float(loss)

    # ---- predicción ----------------------------------------------------------------------
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        x = batch[0] if isinstance(batch, list | tuple) else batch
        return torch.sigmoid(self(x).squeeze(1).float())

    # ---- optimización --------------------------------------------------------------------
    def on_train_epoch_start(self) -> None:
        # Backbone congelado (F2.6): también en modo eval, para que BatchNorm use las
        # estadísticas de ImageNet y no pueda adaptarse a una escala de entrada incorrecta.
        backbone = getattr(self.model, "backbone", None)
        if backbone is not None and not any(p.requires_grad for p in backbone.parameters()):
            backbone.eval()

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.scheduler_name == "none":
            return optimizer
        if self.scheduler_name != "cosine":
            raise ValueError(f"scheduler desconocido: {self.scheduler_name!r} (cosine | none)")
        total_steps = int(self.trainer.estimated_stepping_batches)
        steps_per_epoch = max(1, total_steps // max(1, int(self.trainer.max_epochs or 1)))
        warmup_steps = int(round(self.warmup_epochs * steps_per_epoch))
        scheduler = cosine_with_warmup(optimizer, warmup_steps, total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
