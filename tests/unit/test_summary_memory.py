from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from memory.session_memory import SessionMemory
from memory.summary_memory import SummaryMemory


class DummyPrompt:
    def __or__(self, llm):
        return self

    def invoke(self, payload):
        return SimpleNamespace(
            content='{"summary": "updated summary"}',
        )


def test_should_summarize_based_on_turn_count():
    state = {
        "conversation_turn": SummaryMemory.SUMMARY_TRIGGER_TURN,
    }

    assert SummaryMemory.should_summarize(state)


def test_update_summary_uses_llm_and_returns_summary(monkeypatch):
    state = {
        "messages": [
            HumanMessage(content="Hi"),
            AIMessage(content="Hello"),
        ],
        "conversation_summary": "old summary",
    }

    monkeypatch.setattr(
        "memory.summary_memory.LLMFactory.create_qwen_llm",
        lambda temperature=0.1: object(),
    )

    monkeypatch.setattr(
        "memory.summary_memory.ChatPromptTemplate.from_messages",
        lambda messages: DummyPrompt(),
    )

    monkeypatch.setattr(
        "memory.summary_memory.JSONParser.extract_json",
        lambda content: {"summary": "updated summary"},
    )

    result = SummaryMemory.update_summary(state)

    assert result == "updated summary"


def test_update_summary_returns_existing_summary_on_parse_failure(monkeypatch):
    state = {
        "messages": [
            HumanMessage(content="Hi"),
            AIMessage(content="Hello"),
        ],
        "conversation_summary": "existing summary",
    }

    monkeypatch.setattr(
        "memory.summary_memory.LLMFactory.create_qwen_llm",
        lambda temperature=0.1: object(),
    )

    monkeypatch.setattr(
        "memory.summary_memory.ChatPromptTemplate.from_messages",
        lambda messages: DummyPrompt(),
    )

    def raise_parse(_content):
        raise ValueError("invalid json")

    monkeypatch.setattr(
        "memory.summary_memory.JSONParser.extract_json",
        raise_parse,
    )

    result = SummaryMemory.update_summary(state)

    assert result == "existing summary"
