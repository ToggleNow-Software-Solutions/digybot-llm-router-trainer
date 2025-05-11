from __future__ import annotations
import argparse
from pathlib import Path
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from unsloth.chat_templates import train_on_responses_only
from rich import print

from .config import TrainConfig
from .data_utils import load_sharegpt, format_chat
from .model_utils import load_base, add_lora

def main(cfg_path: str):
    cfg = TrainConfig.load(cfg_path)
    print(f"[bold green]Loaded config:[/bold green] {cfg}")

    model, tokenizer = load_base(cfg.base_model)
    model = add_lora(
        model, cfg.target_modules, cfg.lora_r, cfg.lora_alpha, cfg.random_state
    )

    raw_ds = load_sharegpt(cfg.dataset_path)
    ds = format_chat(raw_ds, tokenizer)

    collator = DataCollatorForSeq2Seq(tokenizer, return_tensors="pt")

    training_args = TrainingArguments(
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        max_steps=cfg.max_steps,
        warmup_steps=cfg.warmup_steps,
        fp16=True,
        logging_steps=1,
        output_dir=cfg.output_dir,
        save_total_limit=2,
        evaluation_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        args=training_args,
        data_collator=collator,
        dataset_num_proc=2,
        packing=False,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    print(f"[bold cyan]Model saved to {cfg.output_dir}[/bold cyan]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine‑tune LLaMA with LoRA")
    parser.add_argument(
        "--config", "-c", default="configs/train.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()
    main(args.config)
