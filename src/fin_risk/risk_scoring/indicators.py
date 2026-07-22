"""Financial risk indicator definitions and computation.

Each indicator maps a raw ratio onto a 0-100 "risk score" (higher = riskier)
via piecewise-linear interpolation over hand-set benchmark points. The
benchmarks are simplified, illustrative thresholds for a prototype system,
not a validated credit model — this mirrors how a first-pass rule-based
scorer is typically built before it gets calibrated against real default
data.

Real public disclosures are frequently incomplete (a news digest rarely
carries the full balance sheet), so every indicator is computed defensively:
if a required field is missing, the indicator is marked unavailable rather
than silently defaulting to zero risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from fin_risk.data.loader import CompanyFinancials, PeriodFinancials


@dataclass
class IndicatorSpec:
    key: str
    label: str
    unit: str
    weight: float
    higher_is_riskier: bool
    compute_fn: Callable[[PeriodFinancials, Optional[PeriodFinancials]], Optional[float]]
    points: list[tuple[float, float]]  # (value, risk_score) control points, value ascending
    plain_explanation: str
    benchmark_note: str


@dataclass
class IndicatorResult:
    key: str
    label: str
    unit: str
    weight: float
    value: Optional[float]
    risk_score: Optional[float]
    available: bool
    plain_explanation: str
    benchmark_note: str

    @property
    def formatted_value(self) -> str:
        if self.value is None:
            return "数据缺失"
        if self.unit == "%":
            return f"{self.value * 100:.1f}%"
        return f"{self.value:.2f}{self.unit}"


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _current_ratio(cur: PeriodFinancials, _prev) -> Optional[float]:
    return _safe_div(cur.current_assets, cur.current_liabilities)


def _quick_ratio(cur: PeriodFinancials, _prev) -> Optional[float]:
    if cur.current_assets is None or cur.inventory is None:
        return None
    return _safe_div(cur.current_assets - cur.inventory, cur.current_liabilities)


def _debt_to_asset(cur: PeriodFinancials, _prev) -> Optional[float]:
    return _safe_div(cur.total_liabilities, cur.total_assets)


def _net_margin(cur: PeriodFinancials, _prev) -> Optional[float]:
    return _safe_div(cur.net_profit, cur.revenue)


def _roe(cur: PeriodFinancials, _prev) -> Optional[float]:
    # ROE is meaningless (and sign-flipped) once equity turns negative or zero:
    # a net loss divided by negative equity reads as a large *positive* return,
    # which would understate risk exactly when the company is most distressed.
    if cur.equity is not None and cur.equity <= 0:
        return None
    return _safe_div(cur.net_profit, cur.equity)


def _ocf_to_debt(cur: PeriodFinancials, _prev) -> Optional[float]:
    return _safe_div(cur.operating_cash_flow, cur.total_liabilities)


def _revenue_growth(cur: PeriodFinancials, prev: Optional[PeriodFinancials]) -> Optional[float]:
    if prev is None:
        return None
    return _safe_div(cur.revenue - prev.revenue if cur.revenue is not None and prev.revenue is not None else None, prev.revenue)


INDICATOR_SPECS: list[IndicatorSpec] = [
    IndicatorSpec(
        key="current_ratio",
        label="流动比率",
        unit="倍",
        weight=0.15,
        higher_is_riskier=False,
        compute_fn=_current_ratio,
        points=[(0.5, 95), (1.0, 75), (1.5, 45), (2.0, 20), (3.0, 8)],
        plain_explanation="衡量公司用短期能变现的资产(比如现金、应收账款)去覆盖一年内到期债务的能力,"
        "可以理解成'家里的活期存款和短期能变现的东西,够不够还这个月到期的账单'。数值越低,短期还款压力越大。",
        benchmark_note="一般认为2倍以上较为稳健,低于1倍需要关注短期流动性压力。",
    ),
    IndicatorSpec(
        key="quick_ratio",
        label="速动比率",
        unit="倍",
        weight=0.10,
        higher_is_riskier=False,
        compute_fn=_quick_ratio,
        points=[(0.2, 95), (0.5, 70), (0.8, 40), (1.2, 15), (2.0, 5)],
        plain_explanation="比流动比率更严格,把不容易快速变现的存货剔除后再看短期偿债能力,"
        "相当于'不算囤积的存货,手头能马上拿出来的钱够不够还债'。",
        benchmark_note="一般认为1倍以上较为稳健,明显低于0.5倍说明速动资产对短期负债的覆盖偏紧。",
    ),
    IndicatorSpec(
        key="debt_to_asset",
        label="资产负债率",
        unit="%",
        weight=0.20,
        higher_is_riskier=True,
        compute_fn=_debt_to_asset,
        points=[(0.3, 10), (0.5, 35), (0.65, 55), (0.8, 80), (0.95, 97)],
        plain_explanation="公司总资产里有多少是靠借债撑起来的,相当于'买房子里贷款占房子总价的比例'。"
        "比例越高,说明公司对债权人的依赖越重,一旦经营波动,偿债压力会放大。",
        benchmark_note="制造业/消费类公司一般50%-60%以内相对稳健,超过80%通常已属偏高杠杆。",
    ),
    IndicatorSpec(
        key="net_margin",
        label="净利润率",
        unit="%",
        weight=0.15,
        higher_is_riskier=False,
        compute_fn=_net_margin,
        points=[(-0.5, 98), (-0.1, 80), (0.0, 60), (0.05, 35), (0.15, 10)],
        plain_explanation="每卖出100元的产品或服务,最终能留下多少利润,直接反映主业的赚钱能力。"
        "如果是负数,说明公司在这个报告期是亏损经营的。",
        benchmark_note="能持续保持5%以上通常算健康,持续为负则说明主业造血能力出了问题。",
    ),
    IndicatorSpec(
        key="roe",
        label="净资产收益率(ROE)",
        unit="%",
        weight=0.10,
        higher_is_riskier=False,
        compute_fn=_roe,
        points=[(-0.5, 97), (-0.05, 75), (0.0, 60), (0.05, 35), (0.15, 10)],
        plain_explanation="股东投入公司的每一块钱本金,这个报告期能带来多少回报,"
        "可以理解成'股东这笔投资的收益率'。",
        benchmark_note="长期能维持在10%以上一般被认为是较优秀的盈利水平,持续为负需要重点关注。",
    ),
    IndicatorSpec(
        key="ocf_to_debt",
        label="经营现金流/总负债",
        unit="%",
        weight=0.20,
        higher_is_riskier=False,
        compute_fn=_ocf_to_debt,
        points=[(-0.1, 95), (-0.02, 80), (0.0, 65), (0.05, 40), (0.15, 20), (0.3, 8)],
        plain_explanation="公司主营业务实际收到的现金,相对于全部债务的比例,反映'靠主业赚的真金白银能不能撑住还债'。"
        "利润表上有利润不代表账上有现金,这个指标更接近公司的真实造血能力。",
        benchmark_note="为负值需要高度关注,说明主业经营没有产生现金流入,偿债主要依赖再融资。",
    ),
    IndicatorSpec(
        key="revenue_growth",
        label="营收环比/同比增长率",
        unit="%",
        weight=0.10,
        higher_is_riskier=False,
        compute_fn=_revenue_growth,
        points=[(-0.5, 90), (-0.2, 70), (0.0, 50), (0.05, 30), (0.15, 10)],
        plain_explanation="公司这一期收入相比上一期是在增长还是萎缩,反映业务景气度和市场需求的变化趋势。",
        benchmark_note="连续下滑通常是经营恶化的早期信号,需要结合行业整体环境判断。",
    ),
]


def _score_from_points(value: float, points: list[tuple[float, float]]) -> float:
    xp = [p[0] for p in points]
    fp = [p[1] for p in points]
    return float(np.clip(np.interp(value, xp, fp), 0, 100))


def compute_indicators(company: CompanyFinancials) -> list[IndicatorResult]:
    latest = company.latest
    previous = company.previous
    results: list[IndicatorResult] = []
    for spec in INDICATOR_SPECS:
        value = spec.compute_fn(latest, previous)
        available = value is not None
        risk_score = _score_from_points(value, spec.points) if available else None
        results.append(
            IndicatorResult(
                key=spec.key,
                label=spec.label,
                unit=spec.unit,
                weight=spec.weight,
                value=value,
                risk_score=risk_score,
                available=available,
                plain_explanation=spec.plain_explanation,
                benchmark_note=spec.benchmark_note,
            )
        )
    return results
