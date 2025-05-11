# pylint: disable=redefined-outer-name
"""Shared pytest fixtures for testing configuration and assets."""

from pathlib import Path
import pytest

from llama_finetune.config import TrainConfig

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"


@pytest.fixture(scope="session")
def tiny_cfg_path_fixture() -> Path:
    """Returns path to the small YAML config used for smoke tests."""
    return ASSETS / "tiny_config.yaml"


@pytest.fixture(scope="session")
def tiny_cfg(
    tiny_cfg_path_fixture,
) -> TrainConfig:
    """Loads the TrainConfig from the test YAML file."""
    return TrainConfig.load(tiny_cfg_path_fixture)
