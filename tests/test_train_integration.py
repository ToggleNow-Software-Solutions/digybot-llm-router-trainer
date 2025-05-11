import subprocess, sys, json
from pathlib import Path

def test_train_smoke(tiny_cfg_path, tmp_path):
    """
    End‑to‑end check: run the CLI for 1 step and confirm
    a model + adapter are saved to output_dir.
    """
    # Redirect output dir to a temp path to avoid clutter
    cfg_text = Path(tiny_cfg_path).read_text()
    new_cfg = json.loads(
        json.dumps(cfg_text)  # quick‑and‑dirty deep copy via JSON string
    )
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
