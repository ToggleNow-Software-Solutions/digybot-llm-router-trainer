"""Integration test: run trainer for 1 step and verify model is saved."""

from pathlib import Path
import subprocess
import sys


def test_train_smoke(tiny_cfg_path, tmp_path):
    """
    End‑to‑end check: run the CLI for 1 step and confirm
    a model + adapter are saved to output_dir.
    """
    # Redirect output dir to a temp path to avoid clutter
    cfg_text = Path(tiny_cfg_path).read_text(encoding="utf-8")
    # We only need to adjust the YAML string—not ideal but trivial and fast:
    cfg_text = cfg_text.replace("tests/output", str(tmp_path / "model"))
    cfg_path = tmp_path / "temp_cfg.yaml"
    cfg_path.write_text(cfg_text)

    # Run the trainer module as a script
    cmd = [sys.executable, "-m", "llama_finetune.train", "-c", str(cfg_path)]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Trainer prints "Model saved to ..." on success
    assert "Model saved to" in completed.stdout
    assert (tmp_path / "model").is_dir()
