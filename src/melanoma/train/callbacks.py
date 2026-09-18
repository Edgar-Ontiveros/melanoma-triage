"""Callbacks de selección de modelo (F2.2): mejor checkpoint y early stopping por AUPRC de
validación, nunca por pérdida."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

log = logging.getLogger(__name__)


class ThroughputMonitor(Callback):
    """Cuello de botella de E/S por época (F3.4): fracción del tiempo de entrenamiento que el
    bucle pasa esperando al DataLoader y utilización media de la GPU (si pynvml está).

    ``history[epoch] = {"train/data_wait_frac", "train/gpu_util", "train/epoch_min"}``.
    """

    def __init__(self) -> None:
        self.history: dict[int, dict[str, float]] = {}
        self._last_end: float | None = None
        self._batch_start = 0.0
        self._wait = 0.0
        self._compute = 0.0
        self._util: list[float] = []
        self._epoch_start = 0.0

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._last_end = None
        self._wait = self._compute = 0.0
        self._util = []
        self._epoch_start = time.perf_counter()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        now = time.perf_counter()
        if self._last_end is not None:
            self._wait += now - self._last_end
        self._batch_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        now = time.perf_counter()
        self._compute += now - self._batch_start
        self._last_end = now
        if torch.cuda.is_available() and batch_idx % 10 == 0:
            try:
                self._util.append(float(torch.cuda.utilization()))
            except Exception:  # sin pynvml
                pass

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        total = self._wait + self._compute
        stats = {
            "train/data_wait_frac": self._wait / total if total else 0.0,
            "train/gpu_util": sum(self._util) / len(self._util) if self._util else float("nan"),
            "train/epoch_min": (time.perf_counter() - self._epoch_start) / 60,
        }
        self.history[int(trainer.current_epoch)] = stats
        pl_module.log_dict(stats, on_step=False, on_epoch=True)
        msg = (
            f"época {trainer.current_epoch}: {stats['train/epoch_min']:.1f} min, espera de datos "
            f"{100 * stats['train/data_wait_frac']:.0f} %, GPU {stats['train/gpu_util']:.0f} %"
        )
        log.info(msg)
        print(f"\n[E/S] {msg}", flush=True)


def build_callbacks(
    checkpoint_dir: Path | str,
    monitor: str,
    mode: str,
    patience: int,
    run_name: str = "run",
) -> tuple[ModelCheckpoint, EarlyStopping, LearningRateMonitor]:
    metric_tag = monitor.replace("/", "_")
    checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"{run_name}-{{epoch:02d}}-{{{monitor}:.4f}}".replace(monitor, metric_tag),
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    early_stopping = EarlyStopping(monitor=monitor, mode=mode, patience=patience, verbose=True)
    return checkpoint, early_stopping, LearningRateMonitor(logging_interval="step")
