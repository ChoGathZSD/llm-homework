"""LoRA fine-tuning for Gemma 3 270M on detailed-prompt -> SVG pairs."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.data_utils import load_jsonl, prompt_from_row, target_from_row  # noqa: E402


@dataclass
class SvgExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


class SvgDataset(Dataset[SvgExample]):
    def __init__(self, path: str | Path, tokenizer: Any, max_length: int):
        self.rows = load_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> SvgExample:
        row = self.rows[index]
        prompt = prompt_from_row(row)
        target = target_from_row(row)
        eos = self.tokenizer.eos_token or ""

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        target_ids = self.tokenizer(target + eos, add_special_tokens=False)["input_ids"]

        if len(prompt_ids) >= self.max_length:
            prompt_ids = prompt_ids[-(self.max_length // 2) :]
        room_for_target = self.max_length - len(prompt_ids)
        if room_for_target <= 0:
            target_ids = []
        elif len(target_ids) > room_for_target:
            target_ids = target_ids[:room_for_target]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)
        return SvgExample(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


class CausalCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, examples: list[SvgExample]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(ex.input_ids) for ex in examples)
        input_ids, attention_mask, labels = [], [], []
        for ex in examples:
            pad_len = max_len - len(ex.input_ids)
            input_ids.append(ex.input_ids + [pad_id] * pad_len)
            attention_mask.append(ex.attention_mask + [0] * pad_len)
            labels.append(ex.labels + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_path(cfg: dict[str, Any]) -> str:
    model_path = cfg["model"]["name_or_path"]
    fallback = cfg["model"].get("local_fallback_path")
    if fallback and Path(fallback).exists():
        return fallback
    return model_path


def dtype_from_config(value: str) -> torch.dtype:
    value = value.lower()
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if value in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {value}")


def make_training_args(cfg: dict[str, Any]) -> TrainingArguments:
    training = cfg["training"]
    kwargs = {
        "output_dir": cfg["paths"]["output_dir"],
        "num_train_epochs": training["num_train_epochs"],
        "per_device_train_batch_size": training["per_device_train_batch_size"],
        "per_device_eval_batch_size": training["per_device_eval_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "learning_rate": training["learning_rate"],
        "warmup_ratio": training["warmup_ratio"],
        "lr_scheduler_type": training["lr_scheduler_type"],
        "logging_steps": training["logging_steps"],
        "save_strategy": training.get("save_strategy", "epoch"),
        "eval_strategy": training.get("eval_strategy", "epoch"),
        "save_total_limit": training["save_total_limit"],
        "load_best_model_at_end": training.get("load_best_model_at_end", True),
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "fp16": training["fp16"],
        "bf16": training.get("bf16", False),
        "gradient_checkpointing": training["gradient_checkpointing"],
        "optim": training["optim"],
        "max_grad_norm": training["max_grad_norm"],
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "seed": training["seed"],
    }
    if training.get("max_steps") is not None:
        kwargs["max_steps"] = training["max_steps"]
    return TrainingArguments(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(cfg["training"]["seed"])

    model_path = resolve_model_path(cfg)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = dtype_from_config(cfg["model"].get("torch_dtype", "float32"))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        torch_dtype=dtype,
        device_map=cfg["model"].get("device_map", "auto"),
        attn_implementation=cfg["model"].get("attn_implementation", "eager"),
    )
    if cfg["training"]["gradient_checkpointing"]:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = SvgDataset(cfg["paths"]["train_jsonl"], tokenizer, cfg["data"]["max_length"])
    eval_dataset = SvgDataset(cfg["paths"]["valid_jsonl"], tokenizer, cfg["data"]["max_length"])

    trainer = Trainer(
        model=model,
        args=make_training_args(cfg),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train()

    adapter_dir = Path(cfg["paths"]["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    with (adapter_dir / "training_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": cfg,
                "train_rows": len(train_dataset),
                "valid_rows": len(eval_dataset),
                "trainer_state": trainer.state.log_history,
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
