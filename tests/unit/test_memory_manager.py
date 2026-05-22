from memory.memory_manager import MemoryManager


def test_update_memory_adds_messages_and_increments_turn(monkeypatch):
    state = {
        "messages": [],
        "conversation_turn": 1,
        "conversation_summary": "",
    }

    monkeypatch.setattr(
        "memory.memory_manager.SummaryMemory.should_summarize",
        lambda state: False,
    )

    MemoryManager.update_memory(
        state,
        user_query="Hello",
        assistant_response="Hi there",
    )

    assert len(state["messages"]) == 2
    assert state["conversation_turn"] == 2
    assert state["conversation_summary"] == ""
