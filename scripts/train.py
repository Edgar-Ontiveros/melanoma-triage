"""Entrenamiento de una corrida (F2).

Uso local o en Kaggle, siempre desde la raíz del repositorio:

    python scripts/train.py +experiment=b1_resnet50_224 train.seed=0
    python scripts/train.py +experiment=prep_b_divide255 hydra.run.dir=/kaggle/working/runs/prep_b

Deja en el directorio de la corrida: ``checkpoints/``, ``metrics.json``, ``curves.json``,
``val_predictions.csv``, ``figures/``, ``summary.md`` y ``config.yaml``. Solo usa train y val.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from melanoma.train.run import run_training

ROOT = Path(__file__).resolve().parents[1]


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    result = run_training(cfg, root=ROOT, run_dir=run_dir)
    m = result["metrics"]
    print(
        f"\n{result['run_name']}: AUROC {m['auroc']:.4f}  AUPRC {m['auprc']:.4f}  "
        f"(prevalencia {m['prevalence']:.4f})  épocas {result['epochs_run']}  "
        f"colapsos {result['collapse_epochs']}\n→ {run_dir}"
    )


if __name__ == "__main__":
    main()
