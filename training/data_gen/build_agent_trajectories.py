"""Builds the Stage-4 agent-SFT dataset: ReAct-style Thought/Action/Observation
trajectories that teach a model to orchestrate `fin_risk.agent.tools` itself,
instead of following the fixed `pipeline.ReportPipeline` flow.

The *expert policy* being imitated is literally `pipeline._build_rag_query` —
the same hand-written heuristic that currently picks the retrieval query
inside `ReportPipeline.generate_report`. Behavior-cloning it into the model
means the model learns to reproduce (and, after Stage 3 RL, potentially
improve on) a decision that is currently hardcoded.

Each Action's Observation is the output of a *real* tool call
(`fin_risk.agent.tools.call_tool` for retrieval; `assessment_to_dict` applied
to a real/synthetic `RiskAssessment` for the indicator step, since synthetic
financial variants don't exist on disk for `call_tool` to read) — not a
fabricated string. Only the Thought lines are templated natural language.

Usage:
    python -m training.data_gen.build_agent_trajectories --n-per-company 15 \
        --out training/datasets/agent_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from fin_risk.agent.tools import TOOL_SCHEMAS, assessment_to_dict, call_tool
from fin_risk.data.loader import list_companies, load_announcements, load_company, load_news
from fin_risk.llm.client import MockLLMClient
from fin_risk.pipeline import _build_rag_query
from fin_risk.prompts.templates import PromptContext
from fin_risk.rag.retriever import TfidfRetriever
from fin_risk.risk_scoring.scorer import assess

from training.data_gen.synthetic_companies import generate_variants, stable_seed_offset

DEFAULT_AS_OF_DATE = "2026-07-05"

_THOUGHT_STEP1 = [
    "需要先获取这家公司的财务风险指标和评分,才能知道要重点分析什么。",
    "第一步应该拿到该公司最新的风险评分和各项指标明细。",
    "先调用风险评估工具,看看这家公司当前的风险状况和主要风险驱动指标。",
]

_THOUGHT_STEP2_TEMPLATE = (
    "已经拿到风险评估结果,主要风险驱动指标是{drivers}。"
    "接下来需要检索与这些风险点相关的公告或新闻作为证据支撑。"
)

_THOUGHT_STEP3 = [
    "已经拿到财务指标和相关证据,信息足够充分,可以撰写最终风险报告了。",
    "证据和指标都已收集完毕,现在整理成结构化报告。",
]

SYSTEM_PROMPT = (
    "你是一名金融风险分析 Agent,可以调用以下工具来完成分析任务,而不是凭空编造数据:\n\n"
    f"{json.dumps(TOOL_SCHEMAS, ensure_ascii=False, indent=2)}\n\n"
    "请按如下格式逐步思考和行动:\n"
    "Thought: <你的推理>\n"
    'Action: <一个 JSON 对象, 形如 {"tool": "工具名", "arguments": {...}}>\n'
    "Observation: <工具返回结果,会由系统提供,你不需要自己编造>\n"
    "...(可以有多轮 Thought/Action/Observation)...\n"
    "最后用 Final: 给出面向用户的最终风险报告,报告需严格按五段式结构:"
    "【一句话结论】【风险评分与等级】【关键指标通俗解读】【主要风险点与证据】【需关注的趋势与免责声明】,"
    "每条风险论断需引用 [证据n] 或明确写出'基于历史财务数据推断',不得编造未提供的证据或数据,"
    "不得给出买入/卖出等投资建议。"
)


def _compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def render_trajectory(
    company_id: str, variant, variant_idx: int, as_of_date: str, top_k_evidence: int, rng: random.Random
) -> dict:
    assessment = assess(variant)
    obs1 = assessment_to_dict(assessment)

    query = _build_rag_query(assessment)
    obs2 = call_tool("retrieve_evidence", {"company_id": company_id, "query": query, "top_k": top_k_evidence})

    driver_labels = "、".join(ind.label for ind in assessment.top_risk_drivers) or "综合财务状况"

    corpus = load_announcements(company_id) + load_news(company_id)
    retriever = TfidfRetriever(corpus)
    evidence = retriever.retrieve(query, company_id=company_id, top_k=top_k_evidence, reference_date=as_of_date)

    # MockLLMClient renders purely from `context`, ignoring system/user text (see
    # fin_risk.llm.client.MockLLMClient.generate) — no rendered prompt is needed here.
    ctx = PromptContext(assessment=assessment, evidence=evidence, as_of_date=as_of_date)
    report_text = MockLLMClient().generate("", "", context=ctx)

    trajectory = (
        f"Thought: {rng.choice(_THOUGHT_STEP1)}\n"
        f'Action: {_compact({"tool": "get_risk_assessment", "arguments": {"company_id": company_id}})}\n'
        f"Observation: {_compact(obs1)}\n"
        f"Thought: {_THOUGHT_STEP2_TEMPLATE.format(drivers=driver_labels)}\n"
        f'Action: {_compact({"tool": "retrieve_evidence", "arguments": {"company_id": company_id, "query": query, "top_k": top_k_evidence}})}\n'
        f"Observation: {_compact(obs2)}\n"
        f"Thought: {rng.choice(_THOUGHT_STEP3)}\n"
        f"Final: {report_text}"
    )

    user_prompt = f"请分析{assessment.name}({assessment.ticker})的风险状况,写一份风险报告。"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": trajectory},
        ],
        "meta": {
            "company_id": company_id,
            "variant_index": variant_idx,
            "is_real": variant_idx == 0,
            "num_tool_calls": 2,
            "risk_level": assessment.risk_level,
            "evidence_count": len(evidence),
        },
    }


def build_records(n_per_company: int, magnitude: float = 0.20, seed: int = 0, top_k_evidence: int = 4, as_of_date: str = DEFAULT_AS_OF_DATE) -> list[dict]:
    records = []
    for company_id in list_companies():
        base_company = load_company(company_id)
        variants = generate_variants(
            base_company, n_per_company, magnitude=magnitude, seed=seed + stable_seed_offset(company_id)
        )
        rng = random.Random(seed + stable_seed_offset(company_id))
        for variant_idx, variant in enumerate(variants):
            records.append(
                render_trajectory(company_id, variant, variant_idx, as_of_date, top_k_evidence, rng)
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-per-company", type=int, default=15)
    parser.add_argument("--magnitude", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k-evidence", type=int, default=4)
    parser.add_argument("--as-of-date", default=DEFAULT_AS_OF_DATE)
    parser.add_argument("--out", default="training/datasets/agent_sft.jsonl")
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
    print(f"Wrote {len(records)} agent trajectory records to {out_path}")


if __name__ == "__main__":
    main()
