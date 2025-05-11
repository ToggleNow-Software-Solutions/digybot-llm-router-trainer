import pytest
from pathlib import Path
from llama_finetune.config import TrainConfig

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

@pytest.fixture(scope="session")
def tiny_cfg_path() -> Path:
    return ASSETS / "tiny_config.yaml"

@pytest.fixture(scope="session")
def tiny_cfg(tiny_cfg_path) -> TrainConfig:
    return TrainConfig.load(tiny_cfg_path)
