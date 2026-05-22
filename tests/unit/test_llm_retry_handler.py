from unittest.mock import MagicMock

from utils.llm_retry_handler import LLMRetryHandler


class FallbackChain:
    def __init__(self):
        self.invoke = MagicMock(return_value="fallback result")


class DummyChain:
    def __init__(self, first_chain):
        self.first = first_chain

    def invoke(self, payload):
        raise Exception("413 payload too large")


def test_invoke_with_fallback_uses_ollama_chain(monkeypatch):
    fallback = FallbackChain()
    first_chain = MagicMock()
    first_chain.__or__.return_value = fallback

    chain = DummyChain(first_chain)

    monkeypatch.setattr(
        "utils.llm_retry_handler.LLMFactory.create_ollama_llm",
        lambda: object(),
    )

    chain.first = first_chain

    result = LLMRetryHandler.invoke_with_fallback(
        chain=chain,
        payload={"foo": "bar"},
    )

    assert result == "fallback result"
    first_chain.__or__.assert_called_once()
