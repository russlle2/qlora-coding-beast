#!/usr/bin/env python3
"""
Direct QLoRA training for Qwen3-Coder-30B-A3B-Instruct, with sample packing.

Same as simple_train.py but uses TRL's SFTTrainer with `packing=True` to
combine short examples into 8K-token sequences. Reduces step count ~10x.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

BASE = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DATA_PATH = "/workspace/data/uncensored_chatml.jsonl"
OUT_DIR = "/workspace/outputs/adapter_uncensored"
HUB_REPO = "russlle2/qwen3-coder-30b-a3b-adapter-uncensored"
SEQ_LEN = 8192


def main() -> None:
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN env var required")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"[packed] tokenizer for {BASE}")
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("[packed] loading model in 4-bit...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    print(f"[packed] loading dataset from {DATA_PATH}")
    ds = load_dataset("json", data_files=DATA_PATH, split="train")
    print(f"[packed] {len(ds)} rows")

    def to_text(example):
        try:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in example["messages"]]
            text = "\n".join(parts)
        return {"text": text}

    ds = ds.map(to_text, remove_columns=ds.column_names, num_proc=4, desc="format")

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    args = SFTConfig(
        output_dir=OUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        logging_steps=5,
        report_to=["tensorboard"],
        push_to_hub=True,
        hub_model_id=HUB_REPO,
        hub_strategy="every_save",
        hub_private_repo=True,
        hub_token=os.environ["HF_TOKEN"],
        dataloader_num_workers=2,
        seed=42,
        ddp_find_unused_parameters=False,
        max_length=SEQ_LEN,
        packing=True,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        peft_config=lora_cfg,
        processing_class=tokenizer,
    )

    print("[packed] starting trainer.train()")
    trainer.train()

    print("[packed] saving final adapter")
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    trainer.push_to_hub(commit_message="Phase 1 final adapter (packed)")
    print("[packed] DONE")


if __name__ == "__main__":
    main()
