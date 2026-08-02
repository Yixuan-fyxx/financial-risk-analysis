"""Rule-based reward / quality checker for generated risk reports.

This is the verifier used in two places in the pipeline:

- Stage 3 (RL/DPO): score K sampled completions for the same prompt, use the
  score to pick a (chosen, rejected) preference pair — a programmatic reward
  in place of human/LLM preference labeling.
- Eval: score checkpoints (base/sft/merged/dpo/agent) on held-out cases to
  produce a comparable quality number across the pipeline.

The checks encode the same guardrails already written into the v4 system
prompt in `fin_risk.prompts.templates` (locked five-section structure,
citations must be grounded in evidence actually provided, missing data must
be acknowledged rather than fabricated, no sensational language, no buy/sell
advice) — the verifier turns those prose instructions into something a
training loop can score automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

REQUIRED_SECTIONS: list[str] = [
    "【一句话结论】",
    "【风险评分与等级】",
    "【关键指标通俗解读】",
    "【主要风险点与证据】",
    "【需关注的趋势与免责声明】",
]

DISCLAIMER_MARKERS = ["不构成投资建议"]

FORBIDDEN_PHRASES = [
    "建议买入", "建议卖出", "建议持有", "建议增持", "建议减持",
    "暴雷", "雷区",
]

MISSING_DATA_MARKERS = ["数据完整度", "数据缺失", "未披露", "暂缺", "不可用", "推断"]

_CITATION_RE = re.compile(r"\[证据(\d+)\]")

# Relative weights of each check in the aggregate score; must sum to 1.0.
_CHECK_WEIGHTS = {
    "has_all_sections": 0.30,
    "citations_in_range": 0.25,
    "disclaimer_present": 0.15,
    "no_forbidden_language": 0.15,
    "missing_data_acknowledged": 0.15,
}


@dataclass
class VerificationResult:
    score: float
    checks: dict[str, bool]
    details: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def _check_sections(text: str) -> tuple[bool, str]:
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    if missing:
        return False, f"缺少章节: {', '.join(missing)}"
    return True, ""


def _check_citations(text: str, evidence_count: int) -> tuple[bool, str]:
    cited = {int(n) for n in _CITATION_RE.findall(text)}
    if not cited:
        return True, ""
    out_of_range = {n for n in cited if n < 1 or n > evidence_count}
    if out_of_range:
        return False, f"引用了不存在的证据编号: {sorted(out_of_range)} (共提供{evidence_count}条证据)"
    return True, ""


def _check_disclaimer(text: str) -> tuple[bool, str]:
    if any(marker in text for marker in DISCLAIMER_MARKERS):
        return True, ""
    return False, "缺少免责声明(未包含'不构成投资建议')"


def _check_forbidden_language(text: str) -> tuple[bool, str]:
    hits = [p for p in FORBIDDEN_PHRASES if p in text]
    if hits:
        return False, f"包含禁用表述: {', '.join(hits)}"
    return True, ""


def _check_missing_data_acknowledged(text: str, data_coverage: float) -> tuple[bool, str]:
    if data_coverage >= 1.0:
        return True, ""
    if any(marker in text for marker in MISSING_DATA_MARKERS):
        return True, ""
    return False, f"数据完整度仅{data_coverage:.0%},但报告未提及任何数据缺失/推断说明"


def verify_report(report_text: str, *, evidence_count: int, data_coverage: float) -> VerificationResult:
    """Scores a generated report against the guardrails from the v4 prompt.

    `evidence_count` is the number of evidence documents actually passed to
    the model for this prompt (used to catch fabricated `[证据n]` citations).
    `data_coverage` is the fraction of indicator weight backed by available
    data for this assessment (used to check missing-data honesty).
    """
    checks: dict[str, bool] = {}
    details: list[str] = []

    for name, fn in [
        ("has_all_sections", lambda: _check_sections(report_text)),
        ("citations_in_range", lambda: _check_citations(report_text, evidence_count)),
        ("disclaimer_present", lambda: _check_disclaimer(report_text)),
        ("no_forbidden_language", lambda: _check_forbidden_language(report_text)),
        ("missing_data_acknowledged", lambda: _check_missing_data_acknowledged(report_text, data_coverage)),
    ]:
        ok, detail = fn()
        checks[name] = ok
        if not ok:
            details.append(detail)

    score = sum(_CHECK_WEIGHTS[name] for name, ok in checks.items() if ok)
    return VerificationResult(score=round(score, 4), checks=checks, details=details)


def is_acceptable(result: VerificationResult, threshold: float = 0.8) -> bool:
    return result.score >= threshold
