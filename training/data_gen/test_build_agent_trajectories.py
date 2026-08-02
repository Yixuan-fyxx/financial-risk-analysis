import json

from training.data_gen.build_agent_trajectories import build_records
from training.verifier.report_verifier import is_acceptable, verify_report


def test_build_records_covers_all_three_companies():
    records = build_records(n_per_company=2, seed=0)
    company_ids = {r["meta"]["company_id"] for r in records}
    assert company_ids == {"000333", "600585", "3333HK"}
    assert len(records) == 6


def test_trajectory_contains_expected_react_structure():
    records = build_records(n_per_company=1, seed=0)
    for rec in records:
        trajectory = rec["messages"][-1]["content"]
        assert trajectory.count("Thought:") == 3
        assert trajectory.count("Action:") == 2
        assert trajectory.count("Observation:") == 2
        assert "Final:" in trajectory
        assert trajectory.index("Action:") < trajectory.index("Observation:")


def test_actions_are_valid_json_matching_known_tools():
    records = build_records(n_per_company=1, seed=0)
    for rec in records:
        trajectory = rec["messages"][-1]["content"]
        action_lines = [line for line in trajectory.splitlines() if line.startswith("Action: ")]
        assert len(action_lines) == 2
        tools_called = []
        for line in action_lines:
            payload = json.loads(line[len("Action: "):])
            assert "tool" in payload and "arguments" in payload
            tools_called.append(payload["tool"])
        assert tools_called == ["get_risk_assessment", "retrieve_evidence"]


def test_observations_are_valid_json():
    records = build_records(n_per_company=1, seed=0)
    for rec in records:
        trajectory = rec["messages"][-1]["content"]
        obs_lines = [line for line in trajectory.splitlines() if line.startswith("Observation: ")]
        for line in obs_lines:
            json.loads(line[len("Observation: "):])  # must not raise


def test_final_report_passes_verifier():
    records = build_records(n_per_company=2, seed=0)
    for rec in records:
        trajectory = rec["messages"][-1]["content"]
        final_text = trajectory.split("Final: ", 1)[1]
        verdict = verify_report(final_text, evidence_count=rec["meta"]["evidence_count"], data_coverage=1.0)
        # data_coverage stand-in of 1.0 here only exercises structure/citation
        # checks (not the missing-data-acknowledgement check).
        assert verdict.checks["has_all_sections"]
        assert verdict.checks["citations_in_range"]


def test_system_prompt_embeds_tool_schemas():
    records = build_records(n_per_company=1, seed=0)
    system_prompt = records[0]["messages"][0]["content"]
    assert "get_risk_assessment" in system_prompt
    assert "retrieve_evidence" in system_prompt


def test_build_records_is_reproducible_given_same_seed():
    a = build_records(n_per_company=2, seed=5)
    b = build_records(n_per_company=2, seed=5)
    assert a == b
