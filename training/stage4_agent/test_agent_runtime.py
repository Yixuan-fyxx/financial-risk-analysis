import json

from training.stage4_agent.agent_runtime import STOP_MARKER, _extract_action, run_agent


def test_extract_action_parses_valid_json_line():
    text = 'Thought: 先看指标\nAction: {"tool": "get_risk_assessment", "arguments": {"company_id": "600585"}}'
    action = _extract_action(text)
    assert action == {"tool": "get_risk_assessment", "arguments": {"company_id": "600585"}}


def test_extract_action_returns_none_when_missing():
    assert _extract_action("Thought: 只是在想\n") is None


def test_extract_action_returns_none_on_malformed_json():
    assert _extract_action("Action: {not valid json}") is None


def test_run_agent_executes_real_tool_call_and_feeds_back_observation():
    turns = iter(
        [
            'Thought: 先获取风险指标\nAction: {"tool": "get_risk_assessment", "arguments": {"company_id": "600585"}}'
            + STOP_MARKER
            + " 我瞎编的观察结果",  # model tries to hallucinate an Observation -> must be cut
            "Thought: 信息足够了\nFinal: 【一句话结论】测试报告,不构成投资建议。",
        ]
    )

    seen_contexts = []

    def fake_generate(context: str) -> str:
        seen_contexts.append(context)
        return next(turns)

    trace = run_agent(fake_generate, "请分析海螺水泥的风险状况,写一份风险报告。", max_turns=4)

    assert "Final: 【一句话结论】" in trace
    assert "我瞎编的观察结果" not in trace  # hallucinated observation must never survive into the trace

    # The real tool result (real company name) must appear in the second call's context,
    # proving the *real* fin_risk.agent.tools.call_tool ran rather than a stub.
    assert any("安徽海螺水泥股份有限公司" in ctx for ctx in seen_contexts[1:])


def test_run_agent_handles_unparseable_action_gracefully():
    turns = iter(
        [
            "Thought: 乱写\nAction: not json at all" + STOP_MARKER,
            "Thought: 放弃\nFinal: 【一句话结论】无法完成,不构成投资建议。",
        ]
    )

    def fake_generate(context: str) -> str:
        return next(turns)

    trace = run_agent(fake_generate, "请分析海螺水泥的风险状况,写一份风险报告。", max_turns=4)
    assert "error" in trace
    assert "Final:" in trace


def test_run_agent_stops_at_max_turns_without_final():
    def fake_generate(context: str) -> str:
        return (
            'Thought: 一直查\nAction: {"tool": "get_risk_assessment", "arguments": {"company_id": "600585"}}'
            + STOP_MARKER
        )

    trace = run_agent(fake_generate, "请分析海螺水泥的风险状况,写一份风险报告。", max_turns=2)
    assert "达到最大轮数" in trace
    assert trace.count("Observation:") == 2


def test_run_agent_terminal_turn_without_stop_marker_returns_immediately():
    def fake_generate(context: str) -> str:
        return "Thought: 直接给结论\nFinal: 【一句话结论】测试,不构成投资建议。"

    trace = run_agent(fake_generate, "请分析海螺水泥的风险状况,写一份风险报告。", max_turns=4)
    assert trace == "Thought: 直接给结论\nFinal: 【一句话结论】测试,不构成投资建议。"
