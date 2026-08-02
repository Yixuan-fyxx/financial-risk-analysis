"""Builds the Stage-1 SFT dataset: (system, user) -> structured v4 risk report.

Uses the existing rule-based `MockLLMClient` (fin_risk.llm.client) as a
"symbolic teacher": it deterministically renders a v4-structured, guardrail-
compliant report from a `PromptContext`, which is exactly the target format
Stage 1 SFT should teach the base model to produce. Financial figures are
augmented via `synthetic_companies.generate_variants`; RAG evidence always
comes from the real company's real announcements/news corpus.

Output is JSONL, one `{"messages": [...], "meta": {...}}` record per line —
the "messages" field is directly consumable by trl's SFTTrainer /
`tokenizer.apply_chat_template`; "meta" is bookkeeping for dataset analysis,
not fed to the model.

Usage:
    python -m training.data_gen.build_sft_dataset --n-per-company 40 \
        --out training/datasets/sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from fin_risk.data.loader import list_companies, load_announcements, load_company, load_news
from fin_risk.llm.client import MockLLMClient
from fin_risk.pipeline import _build_rag_query
from fin_risk.prompts.templates import PromptContext, get_template
from fin_risk.rag.retriever import TfidfRetriever
from fin_risk.risk_scoring.scorer import assess

from training.data_gen.synthetic_companies import generate_variants, stable_seed_offset

DEFAULT_AS_OF_DATE = "2026-07-05"


def build_records(
    n_per_company: int,
    magnitude: float = 0.20,
    seed: int = 0,
    top_k_evidence: int = 4,
    as_of_date: str = DEFAULT_AS_OF_DATE,
) -> list[dict]:
    records = []
    for company_id in list_companies():
        corpus = load_announcements(company_id) + load_news(company_id)
        retriever = TfidfRetriever(corpus)

        base_company = load_company(company_id)
        variants = generate_variants(
            base_company, n_per_company, magnitude=magnitude, seed=seed + stable_seed_offset(company_id)
        )

        for variant_idx, variant in enumerate(variants):
            assessment = assess(variant)
            query = _build_rag_query(assessment)
            evidence = retriever.retrieve(
                query, company_id=company_id, top_k=top_k_evidence, reference_date=as_of_date
            )

            ctx = PromptContext(assessment=assessment, evidence=evidence, as_of_date=as_of_date)
            template = get_template("v4")
            system_prompt, user_prompt = template.build(ctx)
            report_text = MockLLMClient().generate(system_prompt, user_prompt, context=ctx)

            records.append(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": report_text},
                    ],
                    "meta": {
                        "company_id": company_id,
                        "variant_index": variant_idx,
                        "is_real": variant_idx == 0,
                        "risk_level": assessment.risk_level,
                        "risk_score": assessment.risk_score,
                        "data_coverage": assessment.data_coverage,
                        "evidence_count": len(evidence),
                    },
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-per-company", type=int, default=40, help="Snapshots per company (incl. the real one).")
    parser.add_argument("--magnitude", type=float, default=0.20, help="Max relative perturbation per field.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k-evidence", type=int, default=4)
    parser.add_argument("--as-of-date", default=DEFAULT_AS_OF_DATE)
    parser.add_argument("--out", default="training/datasets/sft.jsonl")
    args = parser.parse_args()

    records = build_records(
        n_per_company=args.n_per_company,
        magnitude=args.magnitude,
        seed=args.seed,
        top_k_evidence=args.top_k_evidence,
        as_of_date=args.as_of_date,
    )
    random.Random(args.seed).shuffle(records)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} SFT records to {out_path}")


if __name__ == "__main__":
    main()
