import pytest

from fin_risk.data.loader import CompanyFinancials, Document, PeriodFinancials
from fin_risk.prompts.templates import PromptContext, get_template, list_versions
from fin_risk.rag.retriever import RetrievalResult
from fin_risk.risk_scoring.scorer import assess


def _sample_context() -> PromptContext:
    company = CompanyFinancials(
        company_id="TEST", name="测试公司", ticker="TEST.SZ", industry="测试行业", unit="亿元",
        periods=[
            PeriodFinancials(
                period="P1", revenue=100, net_profit=-30, total_assets=200, total_liabilities=190,
                equity=10, current_assets=50, current_liabilities=120, inventory=20, cash=5,
                operating_cash_flow=-8, short_term_debt=60,
            )
        ],
    )
    assessment = assess(company)
    doc = Document(
        doc_id="D1", company_id="TEST", date="2026-05-01", source_type="news",
        title="测试公司评级下调", content="评级机构下调测试公司信用评级。", tags=["评级下调"],
    )
    evidence = [RetrievalResult(document=doc, similarity=0.5, recency_weight=0.9, combined_score=0.6)]
    return PromptContext(assessment=assessment, evidence=evidence, as_of_date="2026-07-05")


@pytest.mark.parametrize("version", list_versions())
def test_all_versions_build_without_error(version):
    ctx = _sample_context()
    template = get_template(version)
    system, user = template.build(ctx)
    assert isinstance(system, str) and isinstance(user, str)
    assert ctx.assessment.name in user


def test_v1_has_no_role_framing():
    system, _ = get_template("v1").build(_sample_context())
    assert "资深" not in system


def test_v4_locks_output_structure_and_citation_rules():
    _, user = get_template("v4").build(_sample_context())
    for section in ["一句话结论", "风险评分与等级", "关键指标通俗解读", "主要风险点与证据", "需关注的趋势与免责声明"]:
        assert section in user
    assert "[证据1]" in user or "证据" in user
    assert "不构成投资建议" in user  # instructed as part of the mandatory disclaimer
    assert "免责声明" in user


def test_v4_injects_retrieved_evidence_with_citation_marker():
    _, user = get_template("v4").build(_sample_context())
    assert "[证据1]" in user
    assert "评级下调" in user or "测试公司评级下调" in user


def test_current_version_defaults_to_v4():
    assert get_template().version == "v4"
