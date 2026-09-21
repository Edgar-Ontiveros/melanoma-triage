from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


@pytest.fixture(scope="session")
def root_dir() -> Path:
    return ROOT


@pytest.fixture()
def cfg() -> DictConfig:
    """Composición principal (`configs/config.yaml`)."""
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        return compose(config_name="config")


@pytest.fixture()
def smoke_cfg() -> DictConfig:
    """Composición de la prueba de humo (`configs/smoke.yaml`)."""
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        return compose(config_name="smoke")


@pytest.fixture(scope="session")
def prepare_cfg() -> DictConfig:
    """Composición del pipeline de datos (`configs/prepare.yaml`, grupo `data: isic2020`)."""
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        return compose(config_name="prepare")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-local", action="store_true", default=False, help="corre las pruebas @local"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-local"):
        return
    skip = pytest.mark.skip(reason="prueba local: necesita el checkpoint real (--run-local)")
    for item in items:
        if "local" in item.keywords:
            item.add_marker(skip)
