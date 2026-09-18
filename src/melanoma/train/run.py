"""Orquestación de una corrida de entrenamiento (F2): config → semilla → fábrica → DataModule
→ LightningModule → Trainer → evaluación sobre validación → artefactos.

La usan ``scripts/train.py`` (corridas reales) y ``scripts/smoke_train.py`` (humo): el mismo
camino de código, sin celdas copiadas.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.loggers import CSVLogger, Logger
from omegaconf import DictConfig, OmegaConf

from melanoma.data import datamodule_from_config
from melanoma.eval import (
    assert_no_accuracy,
    bootstrap_patient,
    compute_metrics,
    confusion_table,
    metrics_table,
    plot_all,
)
from melanoma.models import build_model
from melanoma.train.callbacks import build_callbacks
from melanoma.train.module import LitBinaryClassifier
from melanoma.utils import seed_everything

log = logging.getLogger(__name__)


def freeze_backbone(model: torch.nn.Module) -> int:
    """Congela ``model.backbone`` (sin gradiente) y deja entrenable solo la cabeza.

    Returns:
        Número de parámetros congelados.
    """
    frozen = 0
    for p in model.backbone.parameters():
        p.requires_grad_(False)
        frozen += p.numel()
    log.info("backbone congelado: %d parámetros sin gradiente", frozen)
    return frozen


def git_sha(root: Path | str, short: bool = True) -> str:
    """SHA del commit actual (``nogit`` si no hay repositorio)."""
    try:
        args = ["git", "-C", str(root), "rev-parse", "--short" if short else "--verify", "HEAD"]
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def run_name(cfg: DictConfig, image_size: int, sha: str) -> str:
    """Convención F2.7: ``{modelo}-{resolución}-s{semilla}-{sha_corto}``."""
    model = str(cfg.model.backbone).split(".")[0]
    return f"{model}-{image_size}-s{cfg.train.seed}-{sha}"


def resolve_pos_weight(value: Any, dm) -> float:
    if value is None:
        return 1.0
    if isinstance(value, str):
        if value == "auto":
            return float(dm.pos_weight())
        raise ValueError(f"train.pos_weight inválido: {value!r} (null | auto | número)")
    return float(value)


def build_logger(cfg: DictConfig, run_dir: Path, name: str, extra: dict[str, Any]) -> Logger:
    config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(config, dict)
    config.update(extra)
    wcfg = cfg.train.wandb
    if wcfg.enabled:
        from lightning.pytorch.loggers import WandbLogger

        if wcfg.mode == "online" and not os.environ.get("WANDB_API_KEY"):
            log.warning("WANDB_API_KEY no está definida; W&B corre en modo offline")
        return WandbLogger(
            project=wcfg.project,
            entity=wcfg.entity,
            name=name,
            save_dir=str(run_dir),
            mode="offline" if not os.environ.get("WANDB_API_KEY") else wcfg.mode,
            config=config,
            tags=[str(cfg.model.backbone).split(".")[0], f"seed{cfg.train.seed}"],
        )
    logger = CSVLogger(save_dir=str(run_dir), name="csv_logs")
    logger.log_hyperparams(config)
    return logger


def run_training(cfg: DictConfig, root: Path | str, run_dir: Path | str) -> dict[str, Any]:
    """Entrena, evalúa sobre validación y deja los artefactos en ``run_dir``.

    Returns:
        Diccionario con ``metrics``, ``ci``, ``best_checkpoint``, ``run_name`` y
        ``collapse_epochs``.
    """
    root, run_dir = Path(root), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    seed_everything(cfg.train.seed, deterministic=cfg.train.deterministic)

    model, data_config = build_model(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
    )
    log.info("data_config del backbone: %s", data_config)
    if bool(cfg.model.get("freeze_backbone", False)):
        freeze_backbone(model)
    dm = datamodule_from_config(cfg.data, data_config, root=root)
    dm.setup("fit")
    image_size = dm.image_size if dm.image_size is not None else int(data_config["input_size"][-1])
    neg, pos = dm.class_counts()
    pos_weight = resolve_pos_weight(cfg.train.pos_weight, dm)
    sha = git_sha(root)
    name = run_name(cfg, int(image_size), sha)
    split_hashes = dm.split_hashes()
    provenance = {
        "git_sha": sha,
        "git_sha_full": git_sha(root, short=False),
        "split_sha256": split_hashes,
        "data_env": dm.images_dir.as_posix(),
        "data_config": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in data_config.items()
        },
        "pos_weight_used": pos_weight,
        "train_counts": {"negatives": neg, "positives": pos},
        "val_counts": dict(zip(["negatives", "positives"], dm.val_ds.class_counts(), strict=True)),
    }
    log.info(
        "corrida %s: train %d neg / %d pos, pos_weight=%.2f, sampler=%s",
        name,
        neg,
        pos,
        pos_weight,
        dm.sampler,
    )

    lit = LitBinaryClassifier(
        model,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        pos_weight=pos_weight,
        scheduler=cfg.train.scheduler,
        warmup_epochs=cfg.train.warmup_epochs,
        collapse_std_threshold=cfg.train.collapse_std_threshold,
        collapse_auc_tolerance=cfg.train.collapse_auc_tolerance,
    )
    ckpt_dir = (
        Path(cfg.train.checkpoint_dir) if cfg.train.checkpoint_dir else run_dir / "checkpoints"
    )
    checkpoint, early_stopping, lr_monitor = build_callbacks(
        ckpt_dir,
        cfg.train.monitor,
        cfg.train.monitor_mode,
        cfg.train.early_stopping_patience,
        run_name=name,
    )
    logger = build_logger(cfg, run_dir, name, provenance)
    trainer = L.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        deterministic=cfg.train.deterministic,
        logger=logger,
        callbacks=[checkpoint, early_stopping, lr_monitor],
        default_root_dir=str(run_dir),
        enable_progress_bar=bool(cfg.train.progress_bar),
        log_every_n_steps=10,
    )
    if trainer.world_size != 1:
        raise RuntimeError(
            f"la evaluación asume un solo proceso y hay {trainer.world_size}; fijar train.devices=1"
        )
    trainer.fit(lit, datamodule=dm)

    # ---- evaluación sobre validación con el mejor checkpoint -----------------------------
    best = checkpoint.best_model_path or None
    history, collapse_history = lit.epoch_history, lit.collapse_history
    if best:
        lit = LitBinaryClassifier.load_from_checkpoint(best, model=model)
        lit.epoch_history, lit.collapse_history = history, collapse_history
    batches = trainer.predict(lit, dataloaders=dm.val_dataloader())
    probs = torch.cat(batches).numpy().astype(np.float64)
    frame = dm.val_frame()
    frame["prob"] = probs
    ev = cfg.train.eval
    metrics = compute_metrics(
        probs,
        frame["target"].to_numpy(),
        frame["patient_id"].to_numpy(),
        fixed_levels=list(ev.fixed_levels),
        threshold_sensitivity=ev.threshold_sensitivity,
        n_reliability_bins=ev.reliability_bins,
    )
    ci = bootstrap_patient(
        probs,
        frame["target"].to_numpy(),
        frame["patient_id"].to_numpy(),
        n_resamples=ev.bootstrap_resamples,
        seed=ev.bootstrap_seed,
    )
    assert_no_accuracy(metrics)
    figures = plot_all(metrics, run_dir / "figures", name)
    collapse_epochs = [i for i, r in enumerate(lit.collapse_history) if r.collapsed]

    frame.to_csv(run_dir / "val_predictions.csv", index=False)
    summary = {
        "run_name": name,
        "best_checkpoint": best,
        "best_epoch": _best_epoch(lit.epoch_history, cfg.train.monitor, cfg.train.monitor_mode),
        "epochs_run": trainer.current_epoch,
        "elapsed_s": round(time.perf_counter() - start, 1),
        "collapse_epochs": collapse_epochs,
        "collapse_history": [r.as_dict() for r in lit.collapse_history],
        "epochs": lit.epoch_history,
        "provenance": provenance,
        "metrics": {k: v for k, v in metrics.items() if k != "curves"},
        "ci": ci,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (run_dir / "curves.json").write_text(json.dumps(metrics["curves"]), encoding="utf-8")
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    (run_dir / "summary.md").write_text(
        f"# {name}\n\ncommit `{sha}` · splits {split_hashes}\n\n"
        + metrics_table(metrics, ci)
        + "\n\n"
        + confusion_table(metrics)
        + (f"\n\n**COLAPSO** en épocas {collapse_epochs}\n" if collapse_epochs else "\n"),
        encoding="utf-8",
    )
    _log_final(logger, metrics, ci, figures, collapse_epochs)
    return {**summary, "figures": figures, "val_frame": frame}


def _best_epoch(history: list[dict], monitor: str, mode: str) -> int | None:
    rows = [r for r in history if monitor in r]
    if not rows:
        return None
    pick = max if mode == "max" else min
    return int(pick(rows, key=lambda r: r[monitor])["epoch"])


def _log_final(
    logger: Logger, metrics: dict, ci: dict, figures: dict, collapse_epochs: list
) -> None:
    final = {f"final/{k}": v for k, v in metrics.items() if isinstance(v, int | float)}
    for k, v in ci.items():
        final[f"final/{k}_ci_lo"] = v["lo"]
        final[f"final/{k}_ci_hi"] = v["hi"]
    final["final/collapse_epochs"] = len(collapse_epochs)
    logger.log_metrics(final)
    experiment = getattr(logger, "experiment", None)
    if experiment is not None and hasattr(experiment, "log") and hasattr(experiment, "summary"):
        try:
            import wandb

            experiment.log({f"figures/{k}": wandb.Image(str(p)) for k, p in figures.items()})
            for k, v in final.items():
                experiment.summary[k] = v
        except Exception as exc:  # pragma: no cover - depende de W&B
            log.warning("no se pudieron registrar las figuras en W&B: %s", exc)


def load_predictions(run_dir: Path | str) -> pd.DataFrame:
    return pd.read_csv(
        Path(run_dir) / "val_predictions.csv", dtype={"image_id": str, "patient_id": str}
    )
