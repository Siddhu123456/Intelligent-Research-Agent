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
        Stream workflow execution
        events from LangGraph.
        """

        final_state = None

        async for event in (
            self._graph.astream(
                state,
                stream_mode="updates",
            )
        ):

            for (
                node_name,
                node_state,
            ) in event.items():

                # Save latest state

                final_state = (
                    node_state
                )

                # Stream lightweight event

                yield {
                    "type": "event",
                    "node": node_name,
                    "step": node_state.get(
                        "current_step",
                    ),
                    "workflow_decision": (
                        node_state.get(
                            "workflow_decision",
                        )
                    ),
                    "message": (
                        self._build_event_message(
                            node_name=node_name,
                        )
                    ),
                }

        # Final completed state

        yield {
            "type": "final_state",
            "state": final_state,
        }
                
    @staticmethod
    def _build_event_message(
        node_name: str,
    ) -> str:

        messages = {
            "supervisor":
                "Supervisor evaluating workflow...",

            "context":
                "Processing contextual query...",

            "query_decomposition":
                "Decomposing research query...",

            "retrieval":
                "Retrieving research documents...",

            "analysis":
                "Analyzing retrieved information...",

            "report_generation":
                "Generating research report...",

            "report_refinement":
                "Refining report sections...",

            "report_chat":
                "Generating conversational response...",

            "document_generation":
                "Generating PDF document...",

            "error_recovery":
                "Handling workflow error...",
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