import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from graph.executor import GraphExecutor
from utils.state_factory import StateFactory


def test_graph_executor_run_invokes_graph(monkeypatch):
    dummy_graph = MagicMock()
    dummy_graph.ainvoke = AsyncMock(return_value={"final_report": "ok"})

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
    state = StateFactory.create_initial_state(query="Query")

    result = asyncio.run(executor.run(state["query"]))

    assert result == {"final_report": "ok"}
    dummy_graph.ainvoke.assert_awaited_once()
