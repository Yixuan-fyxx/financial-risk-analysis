from training.data_gen.build_preference_dataset import build_prompts, build_preference_pairs

GOOD_REPORT = """【一句话结论】
测试公司当前风险等级为「中等风险」(评分40.0/100)。

【风险评分与等级】
综合风险评分 40.0/100。

【关键指标通俗解读】
- 指标A: 说明。

【主要风险点与证据】
- 风险点A。[证据1] 佐证。

【需关注的趋势与免责声明】
本报告仅作为分析参考,不构成投资建议。
"""

BAD_REPORT = "这是一份不合格的报告,没有任何结构,也没有免责声明。"


def test_build_prompts_covers_all_three_companies():
    prompts = build_prompts()
    assert {p["company_id"] for p in prompts} == {"000333", "600585", "3333HK"}
    for p in prompts:
        assert p["system_prompt"] and p["user_prompt"]
        assert isinstance(p["evidence_count"], int)


def test_build_preference_pairs_picks_best_and_worst():
    prompts = build_prompts()[:1]
    responses = iter([GOOD_REPORT, BAD_REPORT, GOOD_REPORT, BAD_REPORT])

    def fake_generate(system_prompt, user_prompt):
        return next(responses)

    pairs = build_preference_pairs(prompts, fake_generate, k=4)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == GOOD_REPORT
    assert pairs[0]["rejected"] == BAD_REPORT
    assert pairs[0]["chosen_score"] > pairs[0]["rejected_score"]


def test_identical_completions_are_skipped_no_signal():
    prompts = build_prompts()[:1]

    def fake_generate(system_prompt, user_prompt):
        return GOOD_REPORT

    pairs = build_preference_pairs(prompts, fake_generate, k=3)
    assert pairs == []


def test_min_score_gap_filters_out_close_scores():
    prompts = build_prompts()[:1]
    responses = iter([GOOD_REPORT, GOOD_REPORT.replace("不构成投资建议", "不构成投资建议 ")])

    def fake_generate(system_prompt, user_prompt):
        return next(responses)

    pairs = build_preference_pairs(prompts, fake_generate, k=2, min_score_gap=0.5)
    assert pairs == []


def test_generate_fn_called_k_times_per_prompt():
    prompts = build_prompts()
    call_count = {"n": 0}

    def fake_generate(system_prompt, user_prompt):
        call_count["n"] += 1
        return GOOD_REPORT if call_count["n"] % 2 == 0 else BAD_REPORT

    build_preference_pairs(prompts, fake_generate, k=3)
    assert call_count["n"] == len(prompts) * 3
