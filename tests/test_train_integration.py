"""Integration test: run trainer for 1 step and verify model is saved."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture(scope="session")
def test_train_smoke(tiny_cfg_path, tmp_path):
    cfg_text = Path(tiny_cfg_path).read_text(encoding="utf-8")
    cfg_text = cfg_text.replace("tests/output", str(tmp_path / "model"))
    cfg_path = tmp_path / "temp_cfg.yaml"
    cfg_path.write_text(cfg_text)

    cmd = [sys.executable, "-m", "llama_finetune.train", "-c", str(cfg_path)]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)

    assert "Model saved to" in completed.stdout
    assert (tmp_path / "model").is_dir()
