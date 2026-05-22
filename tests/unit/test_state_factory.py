from state.constants import CurrentStep
from utils.state_factory import StateFactory


def test_create_initial_state_returns_base_state():
    state = StateFactory.create_initial_state(
        query="A research topic",
    )

    assert state["query"] == "A research topic"
    assert state["contextualized_query"] == "A research topic"
    assert state["current_step"] == CurrentStep.START.value
    assert state["retry_count"] == 0
    assert state["max_retries"] == 2
    assert state["low_confidence"] is False
    assert state["workflow_decision"] == "continue"
    assert state["mode"] == "REPORT_GENERATION"
    assert state["session_id"]


def test_create_initial_state_generates_unique_session_ids():
    first_state = StateFactory.create_initial_state(
        query="Query 1",
    )
    second_state = StateFactory.create_initial_state(
        query="Query 2",
    )

    assert first_state["session_id"] != second_state["session_id"]
