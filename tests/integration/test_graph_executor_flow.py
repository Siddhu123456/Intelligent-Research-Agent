import asyncio
from unittest.mock import MagicMock

from graph.executor import GraphExecutor
from state.constants import CurrentStep
from utils.state_factory import StateFactory


class DummyGraph:
    def __init__(self, events, final_state):
        self._events = events
        self._final_state = final_state

    async def astream(self, state, stream_mode="values"):
        for event in self._events:
            yield event

    async def ainvoke(self, state):
        return self._final_state


def _collect_stream_events(executor, state):
    async def _gather():
        return [event async for event in executor.stream_workflow(state)]

    return asyncio.run(_gather())


def test_graph_executor_stream_workflow_emits_expected_steps(monkeypatch):
    """Test GraphExecutor stream_workflow through minimal step transitions."""

    dummy_events = [
        {"current_step": CurrentStep.START.value},
        {"current_step": CurrentStep.CONTEXTUALIZED.value},
        {"current_step": CurrentStep.DECOMPOSED.value},
        {"current_step": CurrentStep.RETRIEVED.value},
        {"current_step": CurrentStep.ANALYZED.value},
        {
            "current_step": CurrentStep.DONE.value,
            "final_report": "workflow completed",
        },
    ]

    dummy_graph = DummyGraph(
        events=dummy_events,
        final_state={
            "session_id": "session_test",
            "current_step": CurrentStep.DONE.value,
            "final_report": "workflow completed",
        },
    )

    monkeypatch.setattr(
        "graph.executor.GraphBuilder.build_graph",
        MagicMock(return_value=dummy_graph),
    )

    mock_embedding = MagicMock()
    mock_embedding.embed_query = MagicMock()
    monkeypatch.setattr(
        "graph.executor.EmbeddingFactory.get_embeddings",
        MagicMock(return_value=mock_embedding),
    )

    mock_reranker = MagicMock()
    mock_reranker.predict = MagicMock()
    monkeypatch.setattr(
        "graph.executor.RerankerFactory.get_reranker",
        MagicMock(return_value=mock_reranker),
    )

    executor = GraphExecutor()
    state = StateFactory.create_initial_state(query="Evaluate integration test workflow")
    events = _collect_stream_events(executor, state)

    assert [event["step"] for event in events if event["type"] == "event"] == [
        CurrentStep.START.value,
        CurrentStep.CONTEXTUALIZED.value,
        CurrentStep.DECOMPOSED.value,
        CurrentStep.RETRIEVED.value,
        CurrentStep.ANALYZED.value,
        CurrentStep.DONE.value,
    ]

    assert events[-1]["type"] == "final_state"
    assert events[-1]["state"]["current_step"] == CurrentStep.DONE.value
    assert events[-1]["state"]["final_report"] == "workflow completed"
