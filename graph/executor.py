from graph.builder import (
    GraphBuilder,
)

from state.schema import (
    ResearchState,
)

from utils.embedding_factory import EmbeddingFactory
from utils.reranker_factory import RerankerFactory
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

        # Build workflow graph

        self._graph = (
            GraphBuilder.build_graph()
        )

        # Warmup embedding model

        embedding_model = (
            EmbeddingFactory
            .get_embeddings()
        )

        embedding_model.embed_query(
            "warmup",
        )

        # Warmup reranker

        reranker = (
            RerankerFactory
            .get_reranker()
        )

        reranker.predict(
            [
                (
                    "warmup query",
                    "warmup document",
                )
            ]
        )

    async def stream_workflow(
        self,
        state: ResearchState,
    ):
        """
        Stream workflow execution events from LangGraph.

        Uses stream_mode="values" so each emission is the full
        accumulated state snapshot. Deduplicates consecutive
        events for the same agent before yielding.
        """

        final_state    = None
        previous_agent = None

        async for state_update in (
            self._graph.astream(
                state,
                stream_mode="values",
            )
        ):

            final_state   = state_update
            current_agent = state_update.get(
                "current_agent", ""
            )

            # Skip duplicate consecutive agent events
            if current_agent == previous_agent:
                continue

            previous_agent = current_agent

            yield {
                "type":    "event",
                "agent":   current_agent,
                "message": self._build_event_message(
                    current_agent
                ),
            }

        yield {
            "type":  "final_state",
            "state": final_state,
        }

    @staticmethod
    def _build_event_message(
        node_name: str,
    ) -> str:

        messages = {
            "supervisor_agent": (
                "Supervisor evaluating "
                "workflow..."
            ),
            "context_agent": (
                "Processing contextual "
                "query..."
            ),
            "query_decomposition_agent": (
                "Decomposing research "
                "query..."
            ),
            "retrieval_agent": (
                "Retrieving research "
                "documents..."
            ),
            "analysis_agent": (
                "Analyzing retrieved "
                "information..."
            ),
            "report_generation_agent": (
                "Generating research "
                "report..."
            ),
            "report_refinement_agent": (
                "Refining report "
                "sections..."
            ),
            "report_chat_agent": (
                "Generating conversational "
                "response..."
            ),
            "document_generation_agent": (
                "Generating PDF "
                "document..."
            ),
            "error_recovery_agent": (
                "Handling workflow "
                "error..."
            ),
        }

        return messages.get(
            node_name,
            "Executing workflow...",
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