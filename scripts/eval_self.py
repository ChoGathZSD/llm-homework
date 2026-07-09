"""Self-evaluation for base vs LoRA SVG-logo generation."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reward import extract_svg, score_svg  # noqa: E402
from scripts.data_utils import load_jsonl, prompt_from_row, user_prompt_from_row  # noqa: E402


class StopOnTokenSequence(StoppingCriteria):
    def __init__(self, stop_ids: list[int]):
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if not self.stop_ids or input_ids.shape[-1] < len(self.stop_ids):
            return False
        tail = input_ids[0, -len(self.stop_ids) :].tolist()
        return tail == self.stop_ids


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


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


def model_device(model: torch.nn.Module) -> torch.device:
    return next(param.device for param in model.parameters() if param.device.type != "meta")


def load_model_and_tokenizer(cfg: dict[str, Any], adapter_dir: str | None = None) -> tuple[Any, Any]:
    model_path = resolve_model_path(cfg)
    tokenizer_path = adapter_dir if adapter_dir and Path(adapter_dir, "tokenizer_config.json").exists() else model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = dtype_from_config(cfg["model"].get("torch_dtype", "float32"))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        torch_dtype=dtype,
        device_map=cfg["model"].get("device_map", "auto"),
        attn_implementation=cfg["model"].get("attn_implementation", "eager"),
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    if cfg["model"].get("device_map") is None and torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, tokenizer


def generate_svg(model: Any, tokenizer: Any, prompt: str, cfg: dict[str, Any]) -> str:
    generation = cfg["generation"]
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=generation["prompt_max_length"],
    )
    device = model_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_len = encoded["input_ids"].shape[-1]

    gen_kwargs = {
        "max_new_tokens": generation["max_new_tokens"],
        "do_sample": generation["do_sample"],
        "num_beams": generation["num_beams"],
        "repetition_penalty": generation["repetition_penalty"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if generation.get("max_time_seconds"):
        gen_kwargs["max_time"] = generation["max_time_seconds"]
    stop_ids = tokenizer("</svg>", add_special_tokens=False)["input_ids"]
    if stop_ids:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnTokenSequence(stop_ids)])
    if generation["do_sample"]:
        gen_kwargs["temperature"] = generation.get("temperature", 1.0)
        gen_kwargs["top_p"] = generation.get("top_p", 1.0)
    with torch.inference_mode():
        output = model.generate(**encoded, **gen_kwargs)
    new_tokens = output[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [row["reward"]["score"] for row in rows]
    components: dict[str, list[float]] = {}
    for row in rows:
        for name, value in row["reward"]["components"].items():
            components.setdefault(name, []).append(value)
    return {
        "count": len(rows),
        "mean_score": round(statistics.mean(scores), 4) if scores else 0.0,
        "median_score": round(statistics.median(scores), 4) if scores else 0.0,
        "min_score": round(min(scores), 4) if scores else 0.0,
        "max_score": round(max(scores), 4) if scores else 0.0,
        "component_means": {name: round(statistics.mean(values), 4) for name, values in components.items()},
    }


def evaluate_one(name: str, cfg: dict[str, Any], rows: list[dict[str, Any]], adapter_dir: str | None) -> dict[str, Any]:
    out_dir = Path(cfg["paths"]["outputs_dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(cfg, adapter_dir)

    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        started = datetime.now(timezone.utc)
        print(f"[{name}] {index + 1}/{len(rows)} generating...", flush=True)
        prompt = prompt_from_row(row)
        user_prompt = user_prompt_from_row(row)
        raw = generate_svg(model, tokenizer, prompt, cfg)
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        candidate, had_wrapping_text = extract_svg(raw)
        reward = score_svg(raw, user_prompt)

        stem = f"{index:03d}"
        (out_dir / f"{stem}.raw.txt").write_text(raw, encoding="utf-8")
        (out_dir / f"{stem}.svg").write_text(candidate, encoding="utf-8")
        results.append(
            {
                "index": index,
                "prompt": user_prompt,
                "raw_output_path": str(out_dir / f"{stem}.raw.txt"),
                "svg_path": str(out_dir / f"{stem}.svg"),
                "had_wrapping_text": had_wrapping_text,
                "generation_seconds": round(elapsed, 3),
                "reward": reward,
            }
        )
        print(f"[{name}] {index + 1}/{len(rows)} score={reward['score']:.2f} seconds={elapsed:.1f}", flush=True)

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"summary": summarize(results), "items": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train_config.yaml")
    parser.add_argument("--base-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    rows = load_jsonl(cfg["paths"]["valid_jsonl"])
    output: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "valid_rows": len(rows),
        "models": {},
    }
    output["models"]["base"] = evaluate_one("base", cfg, rows, adapter_dir=None)

    adapter_dir = Path(cfg["paths"]["adapter_dir"])
    if not args.base_only and adapter_dir.exists():
        output["models"]["adapter"] = evaluate_one("adapter", cfg, rows, adapter_dir=str(adapter_dir))
        base_mean = output["models"]["base"]["summary"]["mean_score"]
        tuned_mean = output["models"]["adapter"]["summary"]["mean_score"]
        output["delta"] = {
            "mean_score": round(tuned_mean - base_mean, 4),
            "relative_percent": round((tuned_mean - base_mean) / max(base_mean, 1e-9) * 100, 4),
        }
    elif not args.base_only:
        output["adapter_missing"] = str(adapter_dir)

    results_path = Path(cfg["paths"]["results_json"])
    results_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
