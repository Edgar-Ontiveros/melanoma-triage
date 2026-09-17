"""Prueba de humo de entrenamiento.

Recorre el camino completo config → semilla → fábrica → DataModule real (JPEG sintéticos en
disco, albumentations, sampler) → LightningModule → Trainer → evaluación con bootstrap por
paciente → artefactos. No descarga pesos ni datasets. Es el gate que se corre antes de
encender cualquier GPU.

Uso: ``python scripts/smoke_train.py`` (configuración en ``configs/smoke.yaml``).
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from melanoma.data.synthetic import write_synthetic_dataset
from melanoma.eval import assert_no_accuracy
from melanoma.train.run import run_training

ROOT = Path(__file__).resolve().parents[1]


@hydra.main(config_path="../configs", config_name="smoke", version_base="1.3")
def main(cfg: DictConfig) -> None:
    start = time.perf_counter()
    print(OmegaConf.to_yaml(cfg))
    synthetic_root = (ROOT / cfg.data.paths.local.manifest_path).parent
    write_synthetic_dataset(
        synthetic_root,
        n_images=cfg.data.synthetic_samples,
        positive_rate=cfg.data.synthetic_positive_rate,
        seed=cfg.train.seed,
    )
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    result = run_training(cfg, root=ROOT, run_dir=run_dir)
    assert_no_accuracy(result["metrics"])
    for name in ("metrics.json", "val_predictions.csv", "summary.md"):
        assert (run_dir / name).exists(), f"falta {name}"
    for path in result["figures"].values():
        assert Path(path).exists(), f"falta la figura {path}"
    elapsed = time.perf_counter() - start
    print(
        f"SMOKE OK: {cfg.train.max_epochs} épocas, AUROC {result['metrics']['auroc']:.3f}, "
        f"colapsos {result['collapse_epochs']}, {elapsed:.1f} s"
    )


if __name__ == "__main__":
    main()
