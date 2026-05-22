from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from memory.session_memory import SessionMemory


def build_state():
    return {
        "messages": [],
        "conversation_turn": 1,
    }


def test_add_user_message_stores_message():
    state = build_state()

    SessionMemory.add_user_message(
        state,
        "  Hello world  ",
    )

    assert isinstance(
        state["messages"][-1],
        HumanMessage,
    )
    assert state["messages"][-1].content == "Hello world"


def test_add_ai_message_stores_message():
    state = build_state()

    SessionMemory.add_ai_message(
        state,
        "AI response",
    )

    assert isinstance(
        state["messages"][-1],
        AIMessage,
    )
    assert state["messages"][-1].content == "AI response"


def test_increment_turn_increases_counter():
    state = build_state()

    SessionMemory.increment_turn(state)

    assert state["conversation_turn"] == 2


def test_clear_messages_resets_history():
    state = build_state()
    state["messages"].append(HumanMessage(content="hi"))

    SessionMemory.clear_messages(state)

    assert state["messages"] == []
