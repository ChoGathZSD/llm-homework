"""Shared data helpers for the SVG-logo LoRA project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SYSTEM_FALLBACK = (
    "You are an expert logo designer working in clean, scalable vector graphics. "
    "Given a description of a logo's visual elements, output one complete SVG "
    "document only."
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def split_messages(row: dict[str, Any]) -> tuple[str, str, str]:
    messages = row["messages"]
    system = next((m["content"] for m in messages if m.get("role") == "system"), SYSTEM_FALLBACK)
    user = next(m["content"] for m in messages if m.get("role") == "user")
    assistant = next(m["content"] for m in messages if m.get("role") == "assistant")
    return system, user, assistant


def format_prompt(system: str, user: str) -> str:
    return (
        "System:\n"
        f"{system.strip()}\n\n"
        "User:\n"
        f"{user.strip()}\n\n"
        "Assistant:\n"
    )


def prompt_from_row(row: dict[str, Any]) -> str:
    system, user, _ = split_messages(row)
    return format_prompt(system, user)


def target_from_row(row: dict[str, Any]) -> str:
    _, _, assistant = split_messages(row)
    return assistant.strip()


def user_prompt_from_row(row: dict[str, Any]) -> str:
    _, user, _ = split_messages(row)
    return user
