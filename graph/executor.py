from graph.builder import (
    GraphBuilder,
)
from state.schema import (
    ResearchState,
)
from utils.state_factory import (
    StateFactory,
)


class GraphExecutor:
    """Executes workflow graph."""

    def __init__(
        self,
    ) -> None:
        self._graph = (
            GraphBuilder.build_graph()
        )

    async def run(
        self,
        query: str,
    ) -> ResearchState:
        initial_state = (
            StateFactory
            .create_initial_state(
                query=query,
            )
        )

        return await self._graph.ainvoke(
            initial_state,
        )

    async def run_with_state(
        self,
        state: ResearchState,
    ) -> ResearchState:
        return await self._graph.ainvoke(
            state,
        )