"""Aggregates individual indicator risk scores into one composite risk assessment."""

from __future__ import annotations

from dataclasses import dataclass

from fin_risk.config import RISK_BANDS
from fin_risk.data.loader import CompanyFinancials
from fin_risk.risk_scoring.indicators import IndicatorResult, compute_indicators


@dataclass
class RiskAssessment:
    company_id: str
    name: str
    ticker: str
    industry: str
    period: str
    risk_score: float
    risk_level: str
    data_coverage: float  # fraction of total indicator weight backed by available data
    indicators: list[IndicatorResult]
    top_risk_drivers: list[IndicatorResult]


def _risk_level(score: float) -> str:
    for low, high, label in RISK_BANDS:
        if low <= score < high:
            return label
    return RISK_BANDS[-1][2]


def assess(company: CompanyFinancials, top_n_drivers: int = 3) -> RiskAssessment:
    indicators = compute_indicators(company)
    available = [ind for ind in indicators if ind.available]

    total_weight = sum(ind.weight for ind in available)
    if total_weight > 0:
        risk_score = sum(ind.risk_score * ind.weight for ind in available) / total_weight
    else:
        risk_score = 50.0  # no usable data: neutral/unknown, not a false "safe" default

    full_weight = sum(ind.weight for ind in indicators)
    data_coverage = total_weight / full_weight if full_weight else 0.0

    top_drivers = sorted(available, key=lambda ind: ind.risk_score * ind.weight, reverse=True)[:top_n_drivers]

    return RiskAssessment(
        company_id=company.company_id,
        name=company.name,
        ticker=company.ticker,
        industry=company.industry,
        period=company.latest.period,
        risk_score=round(risk_score, 1),
        risk_level=_risk_level(risk_score),
        data_coverage=round(data_coverage, 2),
        indicators=indicators,
        top_risk_drivers=top_drivers,
    )
