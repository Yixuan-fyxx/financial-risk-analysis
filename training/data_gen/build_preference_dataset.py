"""Builds the Stage-3 DPO preference dataset via rejection-sampling against
`report_verifier`, instead of human or LLM-judge preference labeling.

For each held-out (company, as_of_date) prompt, sample K completions from a
trained checkpoint (the Stage-2 merged model, in the intended pipeline
order), score every completion with `report_verifier.verify_report`, and
pair the highest-scoring completion as `chosen` against the lowest-scoring
as `rejected`. This is only meaningful once a real checkpoint exists — the
sampling function takes a `generate_fn: Callable[[str, str], str]` so the
core pairing logic (`build_preference_pairs`) is fully unit-testable on CPU
with a fake/deterministic `generate_fn`, without needing transformers/a GPU
to be installed.

Usage (after Stage 2 produces a merged checkpoint):
    python -m training.data_gen.build_preference_dataset \
        --model-path training/outputs/stage2_merged \
        --k 4 --out training/datasets/dpo_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from fin_risk.data.loader import list_companies, load_announcements, load_company, load_news
from fin_risk.pipeline import _build_rag_query
from fin_risk.prompts.templates import PromptContext, get_template
from fin_risk.rag.retriever import TfidfRetriever
from fin_risk.risk_scoring.scorer import assess

from training.verifier.report_verifier import verify_report

DEFAULT_AS_OF_DATE = "2026-07-05"

GenerateFn = Callable[[str, str], str]


def build_prompts(as_of_date: str = DEFAULT_AS_OF_DATE, top_k_evidence: int = 4) -> list[dict]:
    """One prompt per real company: (system_prompt, user_prompt, evidence_count, data_coverage)."""
    prompts = []
    for company_id in list_companies():
        company = load_company(company_id)
        assessment = assess(company)
        corpus = load_announcements(company_id) + load_news(company_id)
        retriever = TfidfRetriever(corpus)
        query = _build_rag_query(assessment)
        evidence = retriever.retrieve(
            query, company_id=company_id, top_k=top_k_evidence, reference_date=as_of_date
        )
        ctx = PromptContext(assessment=assessment, evidence=evidence, as_of_date=as_of_date)
        system_prompt, user_prompt = get_template("v4").build(ctx)
        prompts.append(
            {
                "company_id": company_id,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "evidence_count": len(evidence),
                "data_coverage": assessment.data_coverage,
            }
        )
    return prompts


def build_preference_pairs(
    prompts: list[dict], generate_fn: GenerateFn, k: int = 4, min_score_gap: float = 0.0
) -> list[dict]:
    """For each prompt, samples `k` completions via `generate_fn`, scores them,
    and emits a (chosen, rejected) pair if the score spread exceeds `min_score_gap`.

    A prompt where all K samples score identically (e.g. the policy is
    deterministic and always produces the same output, or has fully
    converged) yields no usable preference signal and is skipped rather than
    faked into a pair.
    """
    pairs = []
    for prompt in prompts:
        completions = [generate_fn(prompt["system_prompt"], prompt["user_prompt"]) for _ in range(k)]
        scored = [
            (
                text,
                verify_report(
                    text, evidence_count=prompt["evidence_count"], data_coverage=prompt["data_coverage"]
                ).score,
            )
            for text in completions
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        best_text, best_score = scored[0]
        worst_text, worst_score = scored[-1]
        if best_score - worst_score <= min_score_gap:
            continue
        pairs.append(
            {
                "company_id": prompt["company_id"],
                "system_prompt": prompt["system_prompt"],
                "user_prompt": prompt["user_prompt"],
                "chosen": best_text,
                "rejected": worst_text,
                "chosen_score": best_score,
                "rejected_score": worst_score,
            }
        )
    return pairs


def _load_hf_generate_fn(model_path: str, max_new_tokens: int = 900, temperature: float = 0.9) -> GenerateFn:
    """Lazily imports transformers so this module stays importable/testable
    without the training extras installed. Only called from `main()`."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.common.config import patch_tokenizer_extra_special_tokens_list_bug

    patch_tokenizer_extra_special_tokens_list_bug()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

    def generate(system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        output = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)

    return generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, help="HF checkpoint dir to sample completions from.")
    parser.add_argument("--k", type=int, default=4, help="Completions sampled per prompt.")
    parser.add_argument("--min-score-gap", type=float, default=0.15)
    parser.add_argument("--as-of-date", default=DEFAULT_AS_OF_DATE)
    parser.add_argument("--top-k-evidence", type=int, default=4)
    parser.add_argument("--out", default="training/datasets/dpo_pairs.jsonl")
    args = parser.parse_args()

    prompts = build_prompts(as_of_date=args.as_of_date, top_k_evidence=args.top_k_evidence)
    generate_fn = _load_hf_generate_fn(args.model_path)
    pairs = build_preference_pairs(prompts, generate_fn, k=args.k, min_score_gap=args.min_score_gap)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} preference pairs (of {len(prompts)} prompts x k={args.k}) to {out_path}")


if __name__ == "__main__":
    main()
