"""Helpers for loading and modifying base LLaMA models using Unsloth."""

from transformers import PreTrainedTokenizerBase, PreTrainedModel
from unsloth import FastLanguageModel

def load_base(model_name: str, dtype="float16") -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a base LLaMA model and tokenizer using Unsloth."""
    model, tok = FastLanguageModel.from_pretrained(
        model_name,
        dtype=dtype,
        load_in_4bit=True,
    )
    return model, tok

def add_lora(
    model: PreTrainedModel,
    target_modules: list[str],
    r: int,
    lora_alpha: int,
    random_state: int,
):
    """Apply LoRA to the base model with specified target modules and parameters."""
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=random_state,
    )
    return model
