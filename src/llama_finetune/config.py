# pylint: disable=too-many-instance-attributes
"""Config dataclass for loading training parameters from a YAML file."""

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """All config values used during training."""

    run_name: str
    base_model: str
    dataset_path: str
    output_dir: str
    batch_size: int
    grad_accum: int
    learning_rate: float
    max_steps: int
    warmup_steps: int
    max_seq_length: int
    lora_r: int
    lora_alpha: int
    target_modules: list[str]
    random_state: int = 42

    @staticmethod
    def load(path: str | Path) -> "TrainConfig":
        """Load a YAML config file and return a validated dataclass."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return TrainConfig(**data)
