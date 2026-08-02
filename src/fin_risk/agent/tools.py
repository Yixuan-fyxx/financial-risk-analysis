"""Wraps fin_risk's business functions as JSON-schema tools.

`pipeline.ReportPipeline` calls `assess()` and `TfidfRetriever.retrieve()`
directly, with the retrieval query hardcoded by `_build_rag_query`. This
module exposes the same underlying operations as named, schema-described
tools instead, so a model can be trained (agent SFT stage) and later run
(`agent_runtime.py`) to decide *itself* when to call each one and what query
to issue — rather than following the fixed orchestration in pipeline.py.

Both training-data generation (expert-demonstration trajectories) and the
live agent runtime dispatch through the same `call_tool()`, so the tool
behavior a model is trained on is exactly the tool behavior it runs against.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from fin_risk.data.loader import load_announcements, load_company, load_news
from fin_risk.rag.retriever import RetrievalResult, TfidfRetriever
from fin_risk.risk_scoring.scorer import RiskAssessment, assess


def assessment_to_dict(a: RiskAssessment) -> dict[str, Any]:
    """Renders a `RiskAssessment` into the JSON shape `get_risk_assessment` returns.

    Factored out so training-data generation can build this same Observation
    payload from an in-memory (possibly synthetic) `RiskAssessment` without
    going back through disk via `get_risk_assessment`.
    """
    return {
        "company_id": a.company_id,
        "name": a.name,
        "ticker": a.ticker,
        "industry": a.industry,
        "period": a.period,
        "risk_score": a.risk_score,
        "risk_level": a.risk_level,
        "data_coverage": a.data_coverage,
        "indicators": [
            {
                "key": ind.key,
                "label": ind.label,
                "value": ind.value,
                "formatted_value": ind.formatted_value,
                "risk_score": ind.risk_score,
                "available": ind.available,
                "weight": ind.weight,
            }
            for ind in a.indicators
        ],
        "top_risk_drivers": [ind.key for ind in a.top_risk_drivers],
    }


def evidence_results_to_dict(results: list[RetrievalResult]) -> dict[str, Any]:
    """Renders retriever results into the JSON shape `retrieve_evidence` returns."""
    return {
        "results": [
            {
                "doc_id": r.document.doc_id,
                "date": r.document.date,
                "source_type": r.document.source_type,
                "title": r.document.title,
                "content": r.document.content,
                "similarity": round(r.similarity, 4),
                "combined_score": round(r.combined_score, 4),
            }
            for r in results
        ]
    }


def get_risk_assessment(company_id: str, as_of_date: Optional[str] = None) -> dict[str, Any]:
    """Compute the current risk assessment for a company from its financial data.

    `as_of_date` is accepted for schema symmetry with `retrieve_evidence` but
    unused: the assessment is always computed from the latest available
    financial period on disk.
    """
    company = load_company(company_id)
    return assessment_to_dict(assess(company))


def retrieve_evidence(
    company_id: str, query: str, top_k: int = 4, as_of_date: Optional[str] = None
) -> dict[str, Any]:
    """Retrieve the most relevant public announcements/news for a company given a query."""
    corpus = load_announcements(company_id) + load_news(company_id)
    retriever = TfidfRetriever(corpus)
    results = retriever.retrieve(query, company_id=company_id, top_k=top_k, reference_date=as_of_date)
    return evidence_results_to_dict(results)


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_risk_assessment",
        "description": (
            "获取指定公司的最新财务风险评分、风险等级、各项指标明细与主要风险驱动指标。"
            "分析一家公司的风险状况时应第一步调用它。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "company_id": {"type": "string", "description": "公司代码,如 '000333'、'600585'、'3333HK'"},
                "as_of_date": {"type": "string", "description": "评估基准日期(可选),格式 YYYY-MM-DD"},
            },
            "required": ["company_id"],
        },
    },
    {
        "name": "retrieve_evidence",
        "description": (
            "检索指定公司相关的企业公告/新闻,用作风险论断的证据支撑。"
            "应在拿到 get_risk_assessment 的结果、明确主要风险驱动指标之后调用;"
            "query 应包含具体的风险指标名称或可能成因关键词,而不是笼统的公司名。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "company_id": {"type": "string", "description": "公司代码"},
                "query": {"type": "string", "description": "检索关键词,建议包含风险指标名称及可能的成因关键词"},
                "top_k": {"type": "integer", "description": "返回结果条数,默认4"},
                "as_of_date": {"type": "string", "description": "检索的时间基准日期(可选),用于时效性加权排序"},
            },
            "required": ["company_id", "query"],
        },
    },
]

TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "get_risk_assessment": get_risk_assessment,
    "retrieve_evidence": retrieve_evidence,
}


class ToolError(Exception):
    """Raised when a tool call fails: unknown tool name, bad arguments, or missing data."""


def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatches a tool call by name against the real fin_risk functions.

    Used both by training-data generation (to produce real Observations for
    expert-demonstration trajectories) and by `agent_runtime.py` (to execute
    tool calls a trained model issues at inference time).
    """
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        raise ToolError(f"Unknown tool: {name!r} (available: {list(TOOL_REGISTRY)})")
    try:
        return fn(**arguments)
    except TypeError as exc:
        raise ToolError(f"Bad arguments for tool {name!r}: {exc}") from exc
    except FileNotFoundError as exc:
        raise ToolError(str(exc)) from exc
