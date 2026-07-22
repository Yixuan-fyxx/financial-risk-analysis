#!/usr/bin/env python
"""CLI demo: generate a risk report for one or all companies in data/companies.

Examples:
    python examples/run_demo.py --list
    python examples/run_demo.py --company 000333
    python examples/run_demo.py --company 3333HK --prompt-version v1
    python examples/run_demo.py --company 600585 --llm anthropic
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from fin_risk.data.loader import list_companies, load_company  # noqa: E402
from fin_risk.pipeline import ReportPipeline  # noqa: E402
from fin_risk.prompts.templates import list_versions  # noqa: E402

DEFAULT_AS_OF_DATE = "2026-07-05"


def print_report(result, company_id: str) -> None:
    a = result.assessment
    print("=" * 78)
    print(f"公司: {a.name} ({a.ticker}) | 行业: {a.industry} | 报告期: {a.period}")
    print(f"综合风险评分: {a.risk_score}/100 | 风险等级: {a.risk_level} | 数据完整度: {a.data_coverage:.0%}")
    print(f"Prompt模板版本: {result.prompt_version} | 检索到证据数: {len(result.evidence)}")
    print("-" * 78)
    print("指标明细:")
    for ind in a.indicators:
        status = f"风险分{ind.risk_score:.0f}" if ind.available else "数据缺失"
        print(f"  - {ind.label:<18} {ind.formatted_value:<12} 权重{ind.weight:>5.0%}  {status}")
    print("-" * 78)
    print("检索到的证据(用于RAG增强):")
    for i, r in enumerate(result.evidence, start=1):
        print(f"  [证据{i}] {r.document.date} | {r.document.title} (相关度{r.similarity:.2f}, 综合分{r.combined_score:.2f})")
    print("-" * 78)
    print("生成的风险分析报告:\n")
    print(result.report_text)
    print("=" * 78)
    print()


def print_prompt(result) -> None:
    print("#" * 78)
    print(f"# 实际发给 LLM 的 system_prompt(版本 {result.prompt_version})")
    print("#" * 78)
    print(result.system_prompt)
    print()
    print("#" * 78)
    print(f"# 实际发给 LLM 的 user_prompt(版本 {result.prompt_version})")
    print("#" * 78)
    print(result.user_prompt)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--company", help="公司ID(如 000333),不指定则运行全部公司")
    parser.add_argument("--llm", default="mock", choices=["mock", "anthropic"], help="LLM客户端类型(默认mock,免费离线)")
    parser.add_argument("--prompt-version", default=None, help=f"Prompt模板版本,可选: {list_versions()}(默认使用当前生产版本)")
    parser.add_argument("--as-of-date", default=DEFAULT_AS_OF_DATE, help="报告生成日期,用于RAG时效性打分与免责声明")
    parser.add_argument("--list", action="store_true", help="列出所有可用公司后退出")
    parser.add_argument("--show-prompt", action="store_true", help="额外打印实际发给LLM的system/user prompt原文,便于调试和调整模板")
    args = parser.parse_args()

    if args.list:
        for cid in list_companies():
            company = load_company(cid)
            print(f"{cid}\t{company.name}\t{company.ticker}\t{company.industry}")
        return

    company_ids = [args.company] if args.company else list_companies()
    pipeline = ReportPipeline(llm_kind=args.llm, prompt_version=args.prompt_version)

    for cid in company_ids:
        result = pipeline.generate_report(cid, as_of_date=args.as_of_date)
        if args.show_prompt:
            print_prompt(result)
        print_report(result, cid)


if __name__ == "__main__":
    main()
