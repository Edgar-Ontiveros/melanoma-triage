from __future__ import annotations

from omegaconf import DictConfig

REQUIRED = {
    "data": {"root", "manifest_path", "splits_dir", "image_size", "batch_size", "num_workers"},
    "model": {"backbone", "pretrained", "dropout", "num_classes"},
    "train": {
        "seed",
        "max_epochs",
        "lr",
        "weight_decay",
        "precision",
        "deterministic",
        "accelerator",
    },
}


def _assert_keys(cfg: DictConfig) -> None:
    for group, keys in REQUIRED.items():
        assert group in cfg, f"falta el grupo {group}"
        missing = keys - set(cfg[group].keys())
        assert not missing, f"faltan claves en {group}: {sorted(missing)}"


def test_config_loads(cfg: DictConfig) -> None:
    _assert_keys(cfg)
    assert isinstance(cfg.model.backbone, str) and cfg.model.backbone
    assert cfg.data.image_size > 0
    assert cfg.train.lr > 0


def test_smoke_config_loads(smoke_cfg: DictConfig) -> None:
    _assert_keys(smoke_cfg)
    assert smoke_cfg.model.pretrained is False, "el humo no debe descargar pesos"
    assert smoke_cfg.train.accelerator == "cpu"
    assert smoke_cfg.data.synthetic_samples > 0
