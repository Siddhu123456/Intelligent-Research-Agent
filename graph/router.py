from langgraph.graph import END

from state.constants import CurrentStep
from state.schema import ResearchState


class GraphRouter:
    """Handles intelligent workflow routing."""

    @staticmethod
    def route(
        state: ResearchState,
    ) -> str:
        if state.get("error"):
            return "error_recovery"

        workflow_decision = (
            state[
                "workflow_decision"
            ]
        )

        if (
            workflow_decision
            == "retry_retrieval"
        ):
            return "retrieval"

        routing_map = {
            CurrentStep.START.value:
                "context",

            CurrentStep.CONTEXTUALIZED.value:
                "query_decomposition",

            CurrentStep.DECOMPOSED.value:
                "retrieval",

            CurrentStep.RETRIEVED.value:
                "analysis",

            CurrentStep.ANALYZED.value:
                "report_generation",

            CurrentStep.DONE.value:
                END,
        }

        return routing_map.get(
            state["current_step"],
            END,
        )