"""Shared scoring helpers for Stage 2's before/after-merge check and the
final `run_eval.py` comparison across all five checkpoints.

Two complementary numbers, computed per checkpoint:
  - general capability: average per-token cross-entropy loss on
    `general_probe.jsonl` (teacher-forced short reference continuations for
    generic, non-finance prompts) — lower is better, and a *rise* after a
    stage is the "did this stage erode general capability" signal.
  - domain quality: average `report_verifier` score on generated risk
    reports for the 3 real companies — higher is better.

Both require transformers/torch and a loaded model — not run as part of the
CPU test suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training.common.chat_format import render_prompt

GENERAL_PROBE_PATH = Path(__file__).parent / "general_probe.jsonl"


def load_general_probes(path: Path = GENERAL_PROBE_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def general_capability_loss(model, tokenizer, probes: list[dict] | None = None) -> float:
    """Average per-token cross-entropy of `reference` given `prompt`, across
    all probes. Lower means the model still assigns high likelihood to
    ordinary, non-finance continuations — i.e. general capability intact."""
    import torch

    probes = probes if probes is not None else load_general_probes()
    losses = []
    for probe in probes:
        prompt_ids = tokenizer(probe["prompt"], return_tensors="pt").input_ids.to(model.device)
        full_ids = tokenizer(probe["prompt"] + probe["reference"], return_tensors="pt").input_ids.to(model.device)
        labels = full_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100  # only score the reference continuation
        with torch.no_grad():
            out = model(full_ids, labels=labels)
        losses.append(out.loss.item())
    return sum(losses) / len(losses)


def domain_verifier_scores(model, tokenizer, prompts: list[dict], max_new_tokens: int = 900) -> list[float]:
    """Generates a report for each domain `prompt` (as built by
    `build_sft_dataset`/`build_preference_dataset`) and scores it with
    `report_verifier.verify_report`. Returns one score per prompt."""
    import torch

    from training.verifier.report_verifier import verify_report

    scores = []
    for prompt in prompts:
        text = render_prompt(prompt["system_prompt"], prompt["user_prompt"])
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        verdict = verify_report(
            completion, evidence_count=prompt["evidence_count"], data_coverage=prompt["data_coverage"]
        )
        scores.append(verdict.score)
    return scores


def summarize(label: str, general_loss: float, domain_scores: list[float]) -> dict[str, Any]:
    return {
        "checkpoint": label,
        "general_capability_loss": round(general_loss, 4),
        "domain_verifier_score_mean": round(sum(domain_scores) / len(domain_scores), 4) if domain_scores else None,
        "domain_verifier_scores": [round(s, 4) for s in domain_scores],
    }
