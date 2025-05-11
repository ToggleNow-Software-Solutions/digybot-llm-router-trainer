"""Unit tests for the data_utils module."""

from transformers import AutoTokenizer
from llama_finetune.data_utils import load_sharegpt, format_chat


def test_sharegpt_load_and_format(tiny_cfg):
    """Ensure dataset loads and chat formatting returns expected fields."""

    # Load a tiny tokenizer to keep speed
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    ds = load_sharegpt(tiny_cfg.dataset_path)
    assert len(ds) == 2

    ds_fmt = format_chat(ds, tok)
    assert "text" in ds_fmt.column_names
    # Now we expect the Inst-style template: <s>[INST] … [/INST] … </s>
    text0 = ds_fmt[0]["text"]
    # should begin with the BOS token + “[INST]”
    assert text0.startswith(f"{tok.bos_token}[INST]"), text0
    # and should end with the EOS token
    assert text0.strip().endswith(tok.eos_token), text0
