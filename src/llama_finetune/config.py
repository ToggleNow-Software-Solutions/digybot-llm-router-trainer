from __future__ import annotations
from dataclasses import dataclass
import yaml
from pathlib import Path

@dataclass(frozen=True, slots=True)
class TrainConfig:
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
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return TrainConfig(**data)
