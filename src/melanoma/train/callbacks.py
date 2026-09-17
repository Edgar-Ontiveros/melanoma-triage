"""Callbacks de selección de modelo (F2.2): mejor checkpoint y early stopping por AUPRC de
validación, nunca por pérdida."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint


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
