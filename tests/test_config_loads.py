from __future__ import annotations

from omegaconf import DictConfig

REQUIRED = {
    "data": {
        "root",
        "manifest_path",
        "splits_dir",
        "image_size",
        "batch_size",
        "num_workers",
        "env",
        "paths",
        "augment",
        "sampler",
        "preprocess_scale",
        "val_resize",
    },
    "model": {"backbone", "pretrained", "dropout", "num_classes"},
    "train": {
        "seed",
        "max_epochs",
        "lr",
        "weight_decay",
        "precision",
        "deterministic",
        "accelerator",
        "devices",
        "pos_weight",
        "scheduler",
        "warmup_epochs",
        "monitor",
        "early_stopping_patience",
        "collapse_std_threshold",
        "collapse_auc_tolerance",
        "eval",
        "wandb",
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


def test_experiment_configs_compose(root_dir) -> None:
    """Cada experimento de configs/experiment/ compone sin error y no toca el split de prueba."""
    from hydra import compose, initialize_config_dir

    exp_dir = root_dir / "configs" / "experiment"
    names = sorted(p.stem for p in exp_dir.glob("*.yaml"))
    assert names, "no hay experimentos"
    with initialize_config_dir(config_dir=str(root_dir / "configs"), version_base="1.3"):
        for name in names:
            c = compose(config_name="config", overrides=[f"+experiment={name}"])
            assert c.train.monitor == "val/auprc", name
            assert c.data.sampler in ("none", "weighted"), name
            assert c.data.preprocess_scale in ("none", "divide255", "multiply255"), name
