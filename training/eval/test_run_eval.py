from training.eval.run_eval import agent_trace_outcome, format_table

GOOD_REPORT = """【一句话结论】
测试。

【风险评分与等级】
测试。

【关键指标通俗解读】
测试。

【主要风险点与证据】
测试。[证据1] 测试。

【需关注的趋势与免责声明】
不构成投资建议。
"""


def test_agent_trace_outcome_completed_with_good_report():
    trace = f"Thought: 测试\nAction: {{}}\nObservation: {{}}\nFinal: {GOOD_REPORT}"
    result = agent_trace_outcome(trace, evidence_count=1)
    assert result["completed"] is True
    assert result["score"] == 1.0


def test_agent_trace_outcome_not_completed_when_max_turns_hit():
    trace = "Thought: 一直查\nAction: {}\nObservation: {}\n[达到最大轮数,未能生成最终报告]"
    result = agent_trace_outcome(trace, evidence_count=1)
    assert result["completed"] is False
    assert result["score"] is None


def test_agent_trace_outcome_completed_but_low_quality_report():
    trace = "Final: 一份不合格的报告,没有结构。"
    result = agent_trace_outcome(trace, evidence_count=1)
    assert result["completed"] is True
    assert result["score"] < 1.0


def test_format_table_includes_all_checkpoint_labels():
    results = [
        {"checkpoint": "base", "general_capability_loss": 1.0, "domain_verifier_score_mean": 0.5,
         "task_completion_rate": None, "agent_domain_score_mean": None},
        {"checkpoint": "stage4_agent", "general_capability_loss": 1.1, "domain_verifier_score_mean": 0.9,
         "task_completion_rate": 1.0, "agent_domain_score_mean": 0.95},
    ]
    table = format_table(results)
    assert "base" in table
    assert "stage4_agent" in table
    assert "checkpoint" in table  # header row present
