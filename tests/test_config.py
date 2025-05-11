from llama_finetune.config import TrainConfig

def test_load_cfg(tiny_cfg):
    assert isinstance(tiny_cfg, TrainConfig)
    assert tiny_cfg.run_name == "test-tiny-run"
    assert tiny_cfg.max_steps == 1
    assert tiny_cfg.batch_size == 1
