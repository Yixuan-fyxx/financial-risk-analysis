from fin_risk.data.loader import CompanyFinancials, PeriodFinancials
from fin_risk.risk_scoring.indicators import compute_indicators


def _company(*periods: PeriodFinancials) -> CompanyFinancials:
    return CompanyFinancials(
        company_id="TEST", name="测试公司", ticker="TEST.SZ", industry="测试行业", unit="亿元",
        periods=list(periods),
    )


def test_healthy_company_gets_low_risk_scores():
    period = PeriodFinancials(
        period="P1", revenue=100, net_profit=15, total_assets=200, total_liabilities=60,
        equity=140, current_assets=90, current_liabilities=30, inventory=10, cash=50,
        operating_cash_flow=20, short_term_debt=10,
    )
    results = {r.key: r for r in compute_indicators(_company(period))}
    assert results["debt_to_asset"].value == 0.3
    assert results["debt_to_asset"].risk_score < 20
    assert results["net_margin"].risk_score < 20
    assert all(r.available for r in results.values() if r.key != "revenue_growth")
    assert results["revenue_growth"].available is False  # no previous period


def test_distressed_company_gets_high_risk_scores():
    period = PeriodFinancials(
        period="P1", revenue=100, net_profit=-80, total_assets=200, total_liabilities=210,
        equity=-10, current_assets=40, current_liabilities=150, inventory=30, cash=2,
        operating_cash_flow=-15, short_term_debt=90,
    )
    results = {r.key: r for r in compute_indicators(_company(period))}
    assert results["debt_to_asset"].risk_score > 90  # liabilities exceed assets
    assert results["net_margin"].risk_score > 90
    assert results["current_ratio"].risk_score > 70


def test_missing_fields_mark_indicator_unavailable_not_zero_risk():
    period = PeriodFinancials(period="P1", revenue=100, net_profit=10, total_assets=None, total_liabilities=None)
    results = {r.key: r for r in compute_indicators(_company(period))}
    assert results["debt_to_asset"].available is False
    assert results["debt_to_asset"].risk_score is None
    assert results["debt_to_asset"].formatted_value == "数据缺失"


def test_roe_is_unavailable_when_equity_is_negative():
    # Net loss / negative equity divides to a false positive ratio — this
    # must be suppressed rather than silently read as a great return.
    period = PeriodFinancials(period="P1", revenue=100, net_profit=-50, equity=-20)
    results = {r.key: r for r in compute_indicators(_company(period))}
    assert results["roe"].available is False


def test_revenue_growth_uses_previous_period():
    p1 = PeriodFinancials(period="P0", revenue=100)
    p2 = PeriodFinancials(period="P1", revenue=110)
    results = {r.key: r for r in compute_indicators(_company(p1, p2))}
    assert abs(results["revenue_growth"].value - 0.10) < 1e-9
