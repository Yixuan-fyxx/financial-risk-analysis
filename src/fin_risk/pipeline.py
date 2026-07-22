"""Orchestrates the end-to-end flow: financials -> risk score -> RAG
retrieval -> prompt rendering -> LLM report generation.
"""

from __future__ import annotations

from dataclasses import dataclass

from fin_risk.data.loader import load_announcements, load_company, load_corpus, load_news
from fin_risk.llm.client import LLMClient, get_llm_client
from fin_risk.prompts.templates import PromptContext, get_template
from fin_risk.rag.retriever import RetrievalResult, TfidfRetriever
from fin_risk.risk_scoring.scorer import RiskAssessment, assess

# Extra keywords per indicator to steer retrieval toward the kind of public
# disclosure that actually explains *why* that indicator is stressed, since
# the indicator label alone ("资产负债率") rarely appears verbatim in news copy.
_DRIVER_QUERY_HINTS = {
    "current_ratio": "流动性 短期偿债 短期借款",
    "quick_ratio": "流动性 短期偿债 存货",
    "debt_to_asset": "负债 杠杆 评级 信用 质押",
    "net_margin": "亏损 利润 减值 盈利能力",
    "roe": "盈利 亏损 股东回报",
    "ocf_to_debt": "现金流 违约 兑付 偿债压力",
    "revenue_growth": "营收下滑 需求 价格战 增长",
}


@dataclass
class ReportResult:
    assessment: RiskAssessment
    evidence: list[RetrievalResult]
    prompt_version: str
    system_prompt: str
    user_prompt: str
    report_text: str


def _build_rag_query(assessment: RiskAssessment) -> str:
    terms = [assessment.name]
    for ind in assessment.top_risk_drivers:
        terms.append(ind.label)
        terms.append(_DRIVER_QUERY_HINTS.get(ind.key, ""))
    return " ".join(t for t in terms if t)


class ReportPipeline:
    def __init__(self, llm_client: LLMClient | None = None, llm_kind: str = "mock", prompt_version: str | None = None):
        self.llm_client = llm_client or get_llm_client(llm_kind)
        self.prompt_version = prompt_version

    def generate_report(self, company_id: str, as_of_date: str, top_k_evidence: int = 4) -> ReportResult:
        company = load_company(company_id)
        assessment = assess(company)

        corpus = load_announcements(company_id) + load_news(company_id)
        retriever = TfidfRetriever(corpus)
        query = _build_rag_query(assessment)
        evidence = retriever.retrieve(
            query, company_id=company_id, top_k=top_k_evidence, reference_date=as_of_date
        )

        ctx = PromptContext(assessment=assessment, evidence=evidence, as_of_date=as_of_date)
        template = get_template(self.prompt_version)
        system_prompt, user_prompt = template.build(ctx)

        report_text = self.llm_client.generate(system_prompt, user_prompt, context=ctx)

        return ReportResult(
            assessment=assessment,
            evidence=evidence,
            prompt_version=template.version,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            report_text=report_text,
        )
