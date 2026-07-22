"""LLM client abstraction: swap between a free offline mock and the real
Anthropic API without touching pipeline code.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

from fin_risk.config import DEFAULT_ANTHROPIC_MODEL
from fin_risk.prompts.templates import PromptContext

# Tags that mark a retrieved document as describing an actual negative/risk
# event, as opposed to routine or positive news. MockLLMClient only cites a
# document against a risk driver when it carries one of these — otherwise it
# says so and falls back to "inferred from financials", rather than force-
# pairing evidence to a claim it doesn't actually support.
_NEGATIVE_SIGNAL_TAGS = {
    "风险提示", "高风险信号", "流动性压力", "评级下调", "诉讼", "股权质押",
    "实际控制人风险", "财务造假", "审计无法表示意见", "清盘令", "清盘呈请",
    "债券违约", "强制退市", "停牌", "延迟披露", "巨额亏损", "营收下滑",
    "营收连续下滑", "资产减值",
}


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, *, context: Optional[PromptContext] = None) -> str:
        """Returns the generated report text for the given system/user prompt.

        `context` carries the same structured data the prompts were rendered
        from. Real clients must ignore it — only `system_prompt`/`user_prompt`
        may reach the actual model. It exists so MockLLMClient can produce a
        realistic, data-faithful demo report without an API call.
        """


class MockLLMClient(LLMClient):
    """Deterministic, offline, zero-cost stand-in for the real LLM.

    It does not parse the rendered prompt text — instead it renders the same
    v4 report structure directly from the structured PromptContext, so demo
    output stays realistic and free even without an API key.
    """

    def generate(self, system_prompt: str, user_prompt: str, *, context: Optional[PromptContext] = None) -> str:
        if context is None:
            return (
                "[MockLLMClient] 未提供结构化上下文,无法生成示例报告。\n"
                f"system_prompt 长度: {len(system_prompt)} 字符, user_prompt 长度: {len(user_prompt)} 字符。"
            )
        return self._render(context)

    def _render(self, ctx: PromptContext) -> str:
        a = ctx.assessment
        lines: list[str] = []

        lines.append("【一句话结论】")
        lead_driver = a.top_risk_drivers[0].label if a.top_risk_drivers else "综合财务状况"
        lines.append(
            f"{a.name}当前风险等级为「{a.risk_level}」(评分{a.risk_score}/100),主要受{lead_driver}影响。"
        )
        lines.append("")

        lines.append("【风险评分与等级】")
        lines.append(
            f"综合风险评分 {a.risk_score}/100,对应风险等级为「{a.risk_level}」"
            f"(评分越高代表风险越高)。本次评估中,有{a.data_coverage:.0%}的指标权重基于可获得的"
            "公开数据计算,其余部分因数据未披露暂缺。"
        )
        lines.append("")

        lines.append("【关键指标通俗解读】")
        top = sorted(
            [ind for ind in a.indicators if ind.available],
            key=lambda ind: ind.risk_score * ind.weight,
            reverse=True,
        )[:5]
        if not top:
            lines.append("暂无足够的公开数据计算关键指标。")
        for ind in top:
            lines.append(
                f"- {ind.label}({ind.formatted_value}): {ind.plain_explanation} "
                f"目前参考基准为「{ind.benchmark_note}」。"
            )
        lines.append("")

        lines.append("【主要风险点与证据】")
        if a.top_risk_drivers:
            used_doc_ids: set[str] = set()
            for ind in a.top_risk_drivers:
                match = next(
                    (
                        (idx, r)
                        for idx, r in enumerate(ctx.evidence, start=1)
                        if r.document.doc_id not in used_doc_ids
                        and set(r.document.tags) & _NEGATIVE_SIGNAL_TAGS
                    ),
                    None,
                )
                if match:
                    idx, r = match
                    used_doc_ids.add(r.document.doc_id)
                    lines.append(
                        f"- {ind.label}处于偏高风险水平({ind.formatted_value})。[证据{idx}] "
                        f"{r.document.date} 「{r.document.title}」与该指标反映的风险方向一致。"
                    )
                else:
                    lines.append(
                        f"- {ind.label}处于偏高风险水平({ind.formatted_value})。"
                        "以下判断基于历史财务数据推断,暂无检索到与之明确对应的外部公开信息佐证。"
                    )
        else:
            lines.append("暂无法识别主要风险驱动指标(数据不足)。")
        lines.append("")

        lines.append("【需关注的趋势与免责声明】")
        lines.append(
            "建议持续关注上述风险指标后续的季度变化趋势,以及是否有新的公开公告或新闻进一步影响评估结论。"
        )
        lines.append(
            f"本报告基于截至{ctx.as_of_date}的公开数据由系统自动生成,仅作为分析参考,不构成投资建议。"
        )
        return "\n".join(lines)


class AnthropicClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, max_tokens: int = 1600):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your environment or .env file, "
                "or use MockLLMClient for offline/free usage."
            )
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "The 'anthropic' package is not installed. Run: pip install anthropic"
            ) from exc

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str, *, context: Optional[PromptContext] = None) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


def get_llm_client(kind: str = "mock", **kwargs) -> LLMClient:
    if kind == "mock":
        return MockLLMClient()
    if kind == "anthropic":
        return AnthropicClient(**kwargs)
    raise ValueError(f"Unknown LLM client kind: {kind!r} (expected 'mock' or 'anthropic')")
