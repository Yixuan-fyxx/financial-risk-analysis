"""Prompt template versions for the risk-report generation step.

Four versions are kept side by side (not just the "current" one) because the
point of this module is the *iteration itself*: each version changes exactly
one dimension — role framing, context organization, or output format — so the
effect of each change stays legible instead of getting bundled into one
opaque rewrite. See docs/prompt_versions.md for example outputs and the
rationale behind each change.

v1 baseline        -> no role, flat number dump, free-form ask
v2 + role           -> adds an explicit persona + lay-audience framing
v3 + context        -> groups indicators by category, surfaces top risk
                       drivers, injects retrieved evidence with citation ids
v4 + output format   -> locks a strict, cited, disclaimer-bearing report
                       structure; tightens the role prompt to force
                       explanations to stay grounded in the supplied
                       canonical indicator definitions instead of the model's
                       own (unverified) financial knowledge
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from fin_risk.rag.retriever import RetrievalResult
from fin_risk.risk_scoring.scorer import RiskAssessment

CATEGORY_MAP = {
    "current_ratio": "偿债能力",
    "quick_ratio": "偿债能力",
    "debt_to_asset": "偿债能力",
    "net_margin": "盈利能力",
    "roe": "盈利能力",
    "ocf_to_debt": "现金流质量",
    "revenue_growth": "成长性",
}


@dataclass
class PromptContext:
    assessment: RiskAssessment
    evidence: list[RetrievalResult]
    as_of_date: str


@dataclass
class PromptVersion:
    version: str
    name: str
    change_notes: list[str]
    build: Callable[[PromptContext], tuple[str, str]]


def _flat_indicator_lines(ctx: PromptContext) -> str:
    return "\n".join(
        f"- {ind.label}: {ind.formatted_value}" for ind in ctx.assessment.indicators
    )


def _grouped_indicator_block(ctx: PromptContext) -> str:
    by_category: dict[str, list] = {}
    for ind in ctx.assessment.indicators:
        by_category.setdefault(CATEGORY_MAP.get(ind.key, "其他"), []).append(ind)
    lines = []
    for category, inds in by_category.items():
        lines.append(f"【{category}】")
        for ind in inds:
            risk_note = f"风险打分{ind.risk_score:.0f}/100" if ind.available else "数据未披露"
            lines.append(
                f"  - {ind.label}({ind.formatted_value}, 权重{ind.weight:.0%}, {risk_note})"
                f" 参考基准: {ind.benchmark_note}"
            )
    return "\n".join(lines)


def _top_drivers_block(ctx: PromptContext) -> str:
    if not ctx.assessment.top_risk_drivers:
        return "(可用数据不足,无法识别主要风险驱动指标)"
    lines = []
    for ind in ctx.assessment.top_risk_drivers:
        lines.append(f"- {ind.label}: {ind.formatted_value} — 规范定义: {ind.plain_explanation}")
    return "\n".join(lines)


def _evidence_block(ctx: PromptContext, with_citation_ids: bool) -> str:
    if not ctx.evidence:
        return "(未检索到相关外部公告或新闻)"
    lines = []
    for i, r in enumerate(ctx.evidence, start=1):
        doc = r.document
        prefix = f"[证据{i}] " if with_citation_ids else ""
        lines.append(
            f"{prefix}{doc.date} | {doc.source_type} | {doc.title}\n    {doc.content}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# v1: baseline — no role, flat context, unconstrained output
# ---------------------------------------------------------------------------

def _build_v1(ctx: PromptContext) -> tuple[str, str]:
    system = "你是一个金融分析助手。"
    a = ctx.assessment
    user = (
        f"公司: {a.name}\n"
        f"风险评分: {a.risk_score}/100\n"
        f"财务指标:\n{_flat_indicator_lines(ctx)}\n\n"
        "请分析这家公司的风险并给出评分。"
    )
    return system, user


# ---------------------------------------------------------------------------
# v2: + role setting — persona + lay-audience framing, context still flat
# ---------------------------------------------------------------------------

def _build_v2(ctx: PromptContext) -> tuple[str, str]:
    system = (
        "你是一名有15年经验的资深金融风险分析师,同时非常擅长把专业内容讲给完全没有金融背景的"
        "普通人听懂。你的读者可能连'资产负债率'是什么都不知道,所以解释专业指标时要先讲清楚"
        "人话含义,再提术语,不要堆砌黑话。"
    )
    a = ctx.assessment
    user = (
        f"公司: {a.name}\n"
        f"风险评分: {a.risk_score}/100\n"
        f"财务指标:\n{_flat_indicator_lines(ctx)}\n\n"
        "请用通俗易懂的语言,向一个完全不懂金融的用户解释这家公司的风险状况。"
    )
    return system, user


# ---------------------------------------------------------------------------
# v3: + context organization — categorized indicators, top risk drivers,
#     retrieved evidence injected (still loose output format)
# ---------------------------------------------------------------------------

def _build_v3(ctx: PromptContext) -> tuple[str, str]:
    system = (
        "你是一名有15年经验的资深金融风险分析师,同时非常擅长把专业内容讲给完全没有金融背景的"
        "普通人听懂。解释专业指标时要先讲清楚人话含义,再提术语,不要堆砌黑话。"
    )
    a = ctx.assessment
    user = (
        f"公司: {a.name}({a.ticker}) | 行业: {a.industry} | 报告期: {a.period}\n"
        f"综合风险评分: {a.risk_score}/100,风险等级: {a.risk_level}\n"
        f"数据完整度: {a.data_coverage:.0%}(部分指标可能因公开披露信息不全而缺失)\n\n"
        f"分类别指标详情:\n{_grouped_indicator_block(ctx)}\n\n"
        f"主要风险驱动指标(按贡献度排序):\n{_top_drivers_block(ctx)}\n\n"
        f"检索到的相关企业公告/新闻:\n{_evidence_block(ctx, with_citation_ids=True)}\n\n"
        "请结合以上财务指标和外部公告/新闻信息,给出通俗易懂的风险解读,并指出主要风险点。"
    )
    return system, user


# ---------------------------------------------------------------------------
# v4 (current/production): + strict output format, tightened role/accuracy
#     constraints so indicator explanations stay grounded in the supplied
#     canonical definitions rather than the model's own unverified priors.
# ---------------------------------------------------------------------------

_V4_OUTPUT_FORMAT = """请严格按以下结构输出报告,使用中文,面向没有金融背景的普通用户:

【一句话结论】
用一句话说清楚:风险等级是什么、最主要的原因是什么。

【风险评分与等级】
给出综合评分、风险等级,并用1-2句话说明这个分数大致意味着什么(不要只报数字)。

【关键指标通俗解读】
逐条列出本次评估中权重最高或风险打分最高的3-5个指标,每条包含:
指标名称 -> 当前数值 -> 用生活化类比讲清楚这个指标衡量什么 -> 当前处于什么水平(参考基准)。
指标含义必须依据下方提供的"规范定义"转述,不要自行编造定义或行业基准。

【主要风险点与证据】
逐条列出主要风险点,每条论断后面用[证据n]标注支撑它的具体公告或新闻编号。
如果某个判断缺乏检索证据支撑,必须明确写出"以下判断基于历史财务数据推断,暂无外部公开信息佐证",
不要编造不存在的证据或事件。

【需关注的趋势与免责声明】
说明后续需要跟踪的变化趋势,并附上免责声明:本报告基于截至{as_of_date}的公开数据自动生成,
仅作为分析参考,不构成投资建议。"""


def _build_v4(ctx: PromptContext) -> tuple[str, str]:
    system = (
        "你是一名服务于普通投资者的资深金融风险分析师,拥有15年信用风险与财务分析经验。"
        "你的核心任务是把专业的财务风险分析,转译成没有金融背景的用户也能看懂的语言。\n"
        "写作要求:\n"
        "1. 先讲人话,再提术语;需要用生活化类比解释专业概念,但类比之后必须回到准确的数据结论。\n"
        "2. 语气客观克制,不夸大风险也不淡化风险,不使用'暴雷''雷区'等煽动性词汇。\n"
        "3. 解释每个指标含义时,必须以用户提供的'规范定义'为准转述,不能凭自己的知识编造定义、"
        "行业基准或未经验证的事实。\n"
        "4. 每一条风险论断都必须有证据支撑:要么引用提供的[证据n],要么明确标注'基于历史财务数据推断'。"
        "严禁编造未被提供的公告、新闻或数据。\n"
        "5. 不得给出买入/卖出/持有等投资建议,只做风险状况的解释与信息整理。"
    )
    a = ctx.assessment
    user = (
        f"<公司概况>\n公司: {a.name}({a.ticker}) | 行业: {a.industry} | 报告期: {a.period}\n"
        f"综合风险评分: {a.risk_score}/100,风险等级: {a.risk_level}\n"
        f"数据完整度: {a.data_coverage:.0%}(低于100%说明部分指标因公开信息未披露相关字段而无法计算)\n"
        "</公司概况>\n\n"
        f"<分类别指标详情>\n{_grouped_indicator_block(ctx)}\n</分类别指标详情>\n\n"
        f"<主要风险驱动指标>\n{_top_drivers_block(ctx)}\n</主要风险驱动指标>\n\n"
        f"<检索到的相关企业公告与新闻>\n{_evidence_block(ctx, with_citation_ids=True)}\n"
        "</检索到的相关企业公告与新闻>\n\n"
        f"{_V4_OUTPUT_FORMAT.format(as_of_date=ctx.as_of_date)}"
    )
    return system, user


TEMPLATE_REGISTRY: dict[str, PromptVersion] = {
    "v1": PromptVersion(
        version="v1",
        name="基线版本",
        change_notes=[
            "无角色设定,模型以默认口吻作答,专业术语解释深浅不受控。",
            "上下文只是把指标名称和数值平铺列出,没有分类、没有权重、没有外部证据。",
            "输出格式完全开放,模型可能给结论也可能只复述数字,长度和结构不稳定。",
        ],
        build=_build_v1,
    ),
    "v2": PromptVersion(
        version="v2",
        name="+ 角色设定",
        change_notes=[
            "新增角色设定:'资深金融风险分析师 + 面向零金融背景用户讲解',显著改善了术语解释的通俗度。",
            "问题:上下文仍是平铺指标,模型经常抓错重点(把不重要的指标讲很细,重点风险一笔带过)。",
            "问题:没有外部信息,模型对'为什么现在风险高'缺乏时效性解释,只能就数字谈数字。",
        ],
        build=_build_v2,
    ),
    "v3": PromptVersion(
        version="v3",
        name="+ 上下文组织(指标分类 + 风险驱动排序 + RAG证据)",
        change_notes=[
            "按偿债能力/盈利能力/现金流质量/成长性对指标分组,并显式标出'主要风险驱动指标',"
            "模型的分析重点明显更聚焦。",
            "注入RAG检索到的企业公告/新闻作为证据,分析开始具备时效性(能引用最近的评级、诉讼、"
            "股权质押等事件),而不仅仅依赖静态财务数字。",
            "问题:输出格式仍是自由发挥,证据引用不规范(有时引用、有时不引用),报告长度、结构在"
            "多次调用之间不稳定,难以稳定解析或做前端展示。",
        ],
        build=_build_v3,
    ),
    "v4": PromptVersion(
        version="v4",
        name="+ 输出格式约束 + 准确性护栏(当前生产版本)",
        change_notes=[
            "锁定五段式输出结构(结论/评分/指标解读/风险点与证据/趋势与免责声明),报告结构稳定,"
            "便于前端渲染和多次生成结果之间的一致性比较。",
            "强制要求指标解释'以提供的规范定义为准转述',不允许模型用自己的知识编造定义或行业基准,"
            "直接针对'风险指标解释的准确性'这个问题。",
            "强制每条风险论断标注证据编号或'基于历史数据推断',减少模型编造未披露事实/证据的风险,"
            "同时提高报告的可解释性(用户能看到结论的证据来源)。",
            "加入免责声明与不得给投资建议的约束,控制模型输出的合规风险。",
        ],
        build=_build_v4,
    ),
}

CURRENT_VERSION = "v4"


def get_template(version: str | None = None) -> PromptVersion:
    return TEMPLATE_REGISTRY[version or CURRENT_VERSION]


def list_versions() -> list[str]:
    return list(TEMPLATE_REGISTRY.keys())
