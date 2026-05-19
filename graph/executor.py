from graph.builder import (
    GraphBuilder,
)

from state.schema import (
    ResearchState,
)

from utils.state_factory import (
    StateFactory,
)

from utils.state_manager import (
    StateManager,
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
        """Run workflow from fresh query."""

        initial_state = (
            StateFactory
            .create_initial_state(
                query=query,
            )
        )

        return await (
            self._graph.ainvoke(
                initial_state,
            )
        )

    async def run_with_state(
        self,
        state: ResearchState,
    ) -> ResearchState:
        """Run workflow with existing state."""

        return await (
            self._graph.ainvoke(
                state,
            )
        )

    async def continue_workflow(
        self,
        previous_state: ResearchState,
        query: str,
    ) -> ResearchState:
        """Continue conversational workflow."""

        prepared_state = (
            StateManager
            .prepare_next_workflow(
                previous_state=(
                    previous_state
                ),
                query=query,
            )
        )

        return await (
            self._graph.ainvoke(
                prepared_state,
            )
        )