"""Shared chat formatting used by every training stage and by `agent_runtime.py`.

The base model (`Qwen2.5-1.5B`, non-Instruct) ships with no chat template —
teaching it one is literally Stage 1 SFT's job. Every stage after that
(merge, DPO, agent SFT) and inference must keep using the exact same
`<|role|>` formatting Stage 1 trained on; if a later stage silently shifted
format, it would be fine-tuning on a distribution the model never learned in
Stage 1. Centralizing it here is what keeps that consistent.
"""

from __future__ import annotations

ROLE_TAGS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}


def render_messages(messages: list[dict], eos_token: str = "") -> str:
    """Renders a full `[{"role": ..., "content": ...}, ...]` list (including
    the assistant's turn) into training text, e.g. for Stage 1/Stage 4 SFT."""
    parts = [f"{ROLE_TAGS[m['role']]}\n{m['content'].strip()}" for m in messages]
    text = "\n".join(parts)
    return text + eos_token if eos_token else text


def render_prompt(system_prompt: str, user_prompt: str) -> str:
    """Renders everything up to (not including) the assistant's turn: the
    generation prompt at inference time, and DPO's `prompt` field."""
    return (
        f"{ROLE_TAGS['system']}\n{system_prompt.strip()}\n"
        f"{ROLE_TAGS['user']}\n{user_prompt.strip()}\n"
        f"{ROLE_TAGS['assistant']}\n"
    )
