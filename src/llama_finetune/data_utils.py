from __future__ import annotations
from datasets import load_dataset, Dataset
import pandas as pd
from transformers import PreTrainedTokenizerBase

def load_sharegpt(path: str) -> Dataset:
    """Load a ShareGPT‑style JSON list into `Dataset`."""
    raw = load_dataset("json", data_files=path, split="train")
    return raw

def format_chat(ds: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """Apply the chat template so each example becomes a single prompt string."""
    def _map(batch):
        texts = [
            tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for conv in batch["conversations"]
        ]
        return {"text": texts}

    return ds.map(_map, batched=True, remove_columns=ds.column_names)
