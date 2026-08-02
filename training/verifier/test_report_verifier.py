from fin_risk.pipeline import ReportPipeline
from training.verifier.report_verifier import is_acceptable, verify_report

GOOD_REPORT = """【一句话结论】
示例公司当前风险等级为「中等风险」(评分40.0/100),主要受营收下滑影响。

【风险评分与等级】
综合风险评分 40.0/100,对应风险等级为「中等风险」。本次评估中,有60%的指标权重基于可获得的公开数据计算,其余因数据未披露暂缺。

【关键指标通俗解读】
- 营收环比/同比增长率(-9.3%): 反映业务景气度变化。目前参考基准为「连续下滑通常是经营恶化早期信号」。

【主要风险点与证据】
- 营收下滑处于偏高风险水平。[证据1] 2026-03-26「示例公告」与该指标反映的风险方向一致。
- 净利润率因数据缺失,以下判断基于历史财务数据推断,暂无检索到与之明确对应的外部公开信息佐证。

【需关注的趋势与免责声明】
建议持续关注后续季度变化。本报告基于公开数据自动生成,仅作为分析参考,不构成投资建议。
"""


def test_real_pipeline_v4_reports_pass_verification_for_all_companies():
    pipeline = ReportPipeline(llm_kind="mock")
    for company_id in ["000333", "600585", "3333HK"]:
        result = pipeline.generate_report(company_id, as_of_date="2026-07-05")
        verdict = verify_report(
            result.report_text,
            evidence_count=len(result.evidence),
            data_coverage=result.assessment.data_coverage,
        )
        assert is_acceptable(verdict), f"{company_id}: {verdict.details}"


def test_hand_written_good_report_scores_perfectly():
    verdict = verify_report(GOOD_REPORT, evidence_count=2, data_coverage=0.6)
    assert verdict.score == 1.0
    assert verdict.passed


def test_missing_sections_is_penalized():
    broken = GOOD_REPORT.replace("【需关注的趋势与免责声明】", "")
    verdict = verify_report(broken, evidence_count=2, data_coverage=0.6)
    assert not verdict.checks["has_all_sections"]
    assert verdict.score < 1.0


def test_fabricated_citation_out_of_range_is_caught():
    broken = GOOD_REPORT.replace("[证据1]", "[证据9]")
    verdict = verify_report(broken, evidence_count=2, data_coverage=0.6)
    assert not verdict.checks["citations_in_range"]


def test_citation_within_range_passes_even_with_few_evidence():
    verdict = verify_report(GOOD_REPORT, evidence_count=1, data_coverage=0.6)
    assert verdict.checks["citations_in_range"]


def test_missing_disclaimer_is_penalized():
    broken = GOOD_REPORT.replace("仅作为分析参考,不构成投资建议。", "")
    verdict = verify_report(broken, evidence_count=2, data_coverage=0.6)
    assert not verdict.checks["disclaimer_present"]


def test_forbidden_investment_advice_language_is_caught():
    broken = GOOD_REPORT + "\n综合来看,建议买入。"
    verdict = verify_report(broken, evidence_count=2, data_coverage=0.6)
    assert not verdict.checks["no_forbidden_language"]


def test_full_data_coverage_does_not_require_missing_data_marker():
    text = GOOD_REPORT.replace("推断", "").replace("数据缺失", "").replace("未披露", "")
    verdict = verify_report(text, evidence_count=2, data_coverage=1.0)
    assert verdict.checks["missing_data_acknowledged"]


def test_partial_coverage_without_acknowledgement_is_penalized():
    text = (
        "【一句话结论】测试\n【风险评分与等级】测试\n【关键指标通俗解读】测试\n"
        "【主要风险点与证据】测试\n【需关注的趋势与免责声明】不构成投资建议\n"
    )
    verdict = verify_report(text, evidence_count=0, data_coverage=0.5)
    assert not verdict.checks["missing_data_acknowledged"]
