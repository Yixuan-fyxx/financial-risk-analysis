import pytest

from fin_risk.llm.client import AnthropicClient, MockLLMClient, get_llm_client


def test_mock_client_without_context_returns_placeholder():
    text = MockLLMClient().generate("sys", "user")
    assert "MockLLMClient" in text


def test_get_llm_client_factory_returns_mock_by_default():
    assert isinstance(get_llm_client(), MockLLMClient)
    assert isinstance(get_llm_client("mock"), MockLLMClient)


def test_get_llm_client_rejects_unknown_kind():
    with pytest.raises(ValueError):
        get_llm_client("not-a-real-client")


def test_anthropic_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        AnthropicClient(api_key=None)
