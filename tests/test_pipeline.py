from fin_risk.pipeline import ReportPipeline


def test_end_to_end_report_generation_for_each_real_company():
    pipeline = ReportPipeline(llm_kind="mock")
    for company_id in ["000333", "600585", "3333HK"]:
        result = pipeline.generate_report(company_id, as_of_date="2026-07-05")
        assert result.assessment.risk_score is not None
        assert result.report_text
        assert "免责声明" in result.report_text
        assert result.prompt_version == "v4"


def test_evergrande_scores_meaningfully_riskier_than_midea():
    pipeline = ReportPipeline(llm_kind="mock")
    midea = pipeline.generate_report("000333", as_of_date="2026-07-05")
    evergrande = pipeline.generate_report("3333HK", as_of_date="2026-07-05")
    assert evergrande.assessment.risk_score > midea.assessment.risk_score
    assert evergrande.assessment.risk_level == "高风险"


def test_evidence_retrieved_is_scoped_to_the_requested_company():
    pipeline = ReportPipeline(llm_kind="mock")
    result = pipeline.generate_report("600585", as_of_date="2026-07-05")
    assert all(r.document.company_id == "600585" for r in result.evidence)


def test_prompt_version_can_be_overridden():
    pipeline = ReportPipeline(llm_kind="mock", prompt_version="v1")
    result = pipeline.generate_report("000333", as_of_date="2026-07-05")
    assert result.prompt_version == "v1"
