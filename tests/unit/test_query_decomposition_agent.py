from agents.query_decomposition import (
    QueryDecompositionAgent,
)
from utils.state_factory import (
    StateFactory,
)


def test_query_decomposition_agent() -> None:
    state = (
        StateFactory.create_initial_state(
            query="What is quantum computing?",
        )
    )

    updated_state = (
        QueryDecompositionAgent.run(
            state,
        )
    )

    assert (
        updated_state["current_step"]
        == "decomposed"
    )

    assert (
        len(
            updated_state["sub_queries"],
        )
        > 0
    )