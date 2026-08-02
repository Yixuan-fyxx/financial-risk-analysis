import pytest

from fin_risk.agent.tools import (
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    ToolError,
    call_tool,
    get_risk_assessment,
    retrieve_evidence,
)


def test_get_risk_assessment_returns_expected_shape():
    result = get_risk_assessment("000333")
    assert result["company_id"] == "000333"
    assert isinstance(result["risk_score"], float)
    assert result["risk_level"] in {"低风险", "中等风险", "较高风险", "高风险"}
    assert isinstance(result["indicators"], list) and result["indicators"]
    assert all("key" in ind and "available" in ind for ind in result["indicators"])
    assert isinstance(result["top_risk_drivers"], list)


def test_get_risk_assessment_unknown_company_raises():
    with pytest.raises(FileNotFoundError):
        get_risk_assessment("NO_SUCH_COMPANY")


def test_retrieve_evidence_scopes_results_to_company():
    result = retrieve_evidence("600585", query="营收下滑", top_k=3)
    assert "results" in result
    assert all(isinstance(r["doc_id"], str) for r in result["results"])
    assert len(result["results"]) <= 3


def test_retrieve_evidence_empty_query_still_returns_shape():
    result = retrieve_evidence("000333", query="", top_k=2)
    assert "results" in result


def test_call_tool_dispatches_get_risk_assessment():
    result = call_tool("get_risk_assessment", {"company_id": "3333HK"})
    assert result["company_id"] == "3333HK"
    assert result["risk_level"] == "高风险"


def test_call_tool_dispatches_retrieve_evidence():
    result = call_tool("retrieve_evidence", {"company_id": "600585", "query": "评级", "top_k": 2})
    assert "results" in result


def test_call_tool_unknown_tool_raises_tool_error():
    with pytest.raises(ToolError):
        call_tool("no_such_tool", {})


def test_call_tool_bad_arguments_raises_tool_error():
    with pytest.raises(ToolError):
        call_tool("get_risk_assessment", {"unexpected_kwarg": "x"})


def test_tool_schemas_match_registry():
    schema_names = {schema["name"] for schema in TOOL_SCHEMAS}
    assert schema_names == set(TOOL_REGISTRY.keys())
    for schema in TOOL_SCHEMAS:
        assert "description" in schema
        assert schema["parameters"]["type"] == "object"
        assert set(schema["parameters"]["required"]) <= set(schema["parameters"]["properties"].keys())
