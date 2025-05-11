"""Training entry point for fine-tuning LLaMA with LoRA using Unsloth."""

import argparse
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
import rich
from unsloth.chat_templates import train_on_responses_only


from llama_finetune.config import TrainConfig
from llama_finetune.data_utils import load_sharegpt, format_chat
from llama_finetune.model_utils import load_base, add_lora

def main(cfg_path: str):
    """Run training loop using config from YAML."""
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
        output_dir=cfg.output_dir,\
        optim="adamw_8bit",
        report_to="none",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407
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
    rich.print(f"[bold cyan]Model saved to {cfg.output_dir}[/bold cyan]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine‑tune LLaMA with LoRA")
    parser.add_argument(
        "--config", "-c", default="configs/train.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()
    main(args.config)
