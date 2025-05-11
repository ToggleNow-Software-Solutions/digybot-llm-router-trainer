# pylint: disable=redefined-outer-name
"""Shared pytest fixtures for testing configuration and assets."""


from pathlib import Path

import pytest
# from transformers import AutoTokenizer

from llama_finetune.config import TrainConfig
# import llama_finetune.model_utils as mu
# import llama_finetune.train as train_mod


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


# @pytest.fixture(autouse=True)
# def stub_heavy_training(monkeypatch):
#     """
#     Stub out heavy model loading and training to speed up integration tests.
#     """

#     # Stub load_base → return dummy model object + tiny tokenizer
#     def fake_load_base():
#         tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
#         return object(), tokenizer

#     monkeypatch.setattr(mu, "load_base", fake_load_base)

#     # Stub add_lora → identity
#     monkeypatch.setattr(mu, "add_lora", lambda model, *args, **kwargs: model)

#     # Stub SFTTrainer to a dummy trainer
#     class DummyTrainer:
#         def __init__(self, *args, **kwargs):
#             pass

#         def train(self):
#             return

#         def save_model(self, output_dir: str):
#             Path(output_dir).mkdir(parents=True, exist_ok=True)

#     monkeypatch.setattr(train_mod, "SFTTrainer", DummyTrainer)

#     # Stub train_on_responses_only → identity
#     monkeypatch.setattr(
#         train_mod, "train_on_responses_only", lambda trainer, **kwargs: trainer
#     )
