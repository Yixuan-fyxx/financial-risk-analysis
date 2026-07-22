from fin_risk.data.loader import CompanyFinancials, PeriodFinancials
from fin_risk.risk_scoring.scorer import assess


def _company(period: PeriodFinancials) -> CompanyFinancials:
    return CompanyFinancials(
        company_id="TEST", name="测试公司", ticker="TEST.SZ", industry="测试行业", unit="亿元",
        periods=[period],
    )


def test_healthy_company_is_low_risk_band():
    period = PeriodFinancials(
        period="P1", revenue=100, net_profit=15, total_assets=200, total_liabilities=60,
        equity=140, current_assets=90, current_liabilities=30, inventory=10, cash=50,
        operating_cash_flow=20, short_term_debt=10,
    )
    result = assess(_company(period))
    assert result.risk_level == "低风险"
    assert result.risk_score < 25


def test_distressed_company_is_high_risk_band():
    period = PeriodFinancials(
        period="P1", revenue=100, net_profit=-80, total_assets=200, total_liabilities=210,
        equity=-10, current_assets=40, current_liabilities=150, inventory=30, cash=2,
        operating_cash_flow=-15, short_term_debt=90,
    )
    result = assess(_company(period))
    assert result.risk_level == "高风险"
    assert result.risk_score >= 75


def test_data_coverage_reflects_missing_indicators():
    period = PeriodFinancials(period="P1", revenue=100, net_profit=10)
    result = assess(_company(period))
    assert 0 < result.data_coverage < 1


def test_no_usable_data_falls_back_to_neutral_score_not_false_safe():
    period = PeriodFinancials(period="P1")
    result = assess(_company(period))
    assert result.risk_score == 50.0
    assert result.data_coverage == 0.0


def test_top_risk_drivers_are_sorted_by_weighted_contribution():
    period = PeriodFinancials(
        period="P1", revenue=100, net_profit=-80, total_assets=200, total_liabilities=210,
        equity=-10, current_assets=40, current_liabilities=150, inventory=30, cash=2,
        operating_cash_flow=-15, short_term_debt=90,
    )
    result = assess(_company(period), top_n_drivers=3)
    contributions = [ind.risk_score * ind.weight for ind in result.top_risk_drivers]
    assert contributions == sorted(contributions, reverse=True)
