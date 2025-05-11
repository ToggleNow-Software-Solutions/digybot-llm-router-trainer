from llama_finetune.data_utils import load_sharegpt, format_chat
from transformers import AutoTokenizer

def test_sharegpt_load_and_format(tmp_path, tiny_cfg):
    # Load a tiny tokenizer to keep speed
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    ds = load_sharegpt(tiny_cfg.dataset_path)
    assert len(ds) == 2

    ds_fmt = format_chat(ds, tok)
    assert "text" in ds_fmt.column_names
    # Each formatted string must start with the user header
    assert ds_fmt[0]["text"].startswith("<|start_header_id|>user")
