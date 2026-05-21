#!/usr/bin/env python3
"""
Direct QLoRA training for Qwen3-Coder-30B-A3B-Instruct without axolotl.

Loads jsonl ChatML dataset → applies tokenizer chat template → trains LoRA on
attention layers in 4-bit → pushes adapter to Hugging Face every save.

Run on the pod:
    cd /workspace/qlora-coding-beast
    python scripts/simple_train.py

Env: HF_TOKEN required (push to hub). WANDB_DISABLED=true recommended.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DATA_PATH = "/workspace/data/uncensored_chatml.jsonl"
OUT_DIR = "/workspace/outputs/adapter_uncensored"
HUB_REPO = "russlle2/qwen3-coder-30b-a3b-adapter-uncensored"
MAX_LEN = 4096


def main() -> None:
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN env var required")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"[simple_train] tokenizer for {BASE}")
    tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    print("[simple_train] loading model in 4-bit...")
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

    print("[simple_train] applying LoRA (attention only)")
    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print(f"[simple_train] loading dataset from {DATA_PATH}")
    ds = load_dataset("json", data_files=DATA_PATH, split="train")
    print(f"[simple_train] {len(ds)} rows")

    def format_and_tokenize(example):
        try:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            parts = []
            for m in example["messages"]:
                parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
            text = "\n".join(parts)
        out = tokenizer(text, truncation=True, max_length=MAX_LEN, padding=False)
        out["labels"] = out["input_ids"].copy()
        return out

    print("[simple_train] tokenizing...")
    ds = ds.map(
        format_and_tokenize,
        remove_columns=ds.column_names,
        num_proc=4,
        desc="tokenize",
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) >= 16, num_proc=4)
    print(f"[simple_train] after filter: {len(ds)} rows")

    args = TrainingArguments(
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
        save_steps=200,
        save_total_limit=3,
        logging_steps=10,
        report_to=["tensorboard"],
        logging_dir=f"{OUT_DIR}/tb",
        push_to_hub=True,
        hub_model_id=HUB_REPO,
        hub_strategy="every_save",
        hub_private_repo=True,
        hub_token=os.environ["HF_TOKEN"],
        dataloader_num_workers=2,
        seed=42,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("[simple_train] starting trainer.train()")
    trainer.train()

    print("[simple_train] saving final adapter")
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    trainer.push_to_hub(commit_message="Phase 1 final adapter")
    print("[simple_train] DONE")


if __name__ == "__main__":
    main()
