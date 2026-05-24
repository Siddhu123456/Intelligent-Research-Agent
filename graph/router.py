from langgraph.graph import (
    END,
)

from state.constants import (
    CurrentStep,
)

from state.schema import (
    ResearchState,
)


class GraphRouter:
    """Handles intelligent workflow routing."""

    @staticmethod
    def route(
        state: ResearchState,
    ) -> str:

        workflow_decision = state.get("workflow_decision", "")

        # Early error handling
        if state.get("error"):
            return "error_recovery"

        # Low-confidence -> retry retrieval
        if state.get("low_confidence"):
            return "retrieval"

        current_step = state.get("current_step", "")

        # Error termination

        if (
            workflow_decision
            == "terminate"
        ):

            return (
                "error_recovery"
            )

        # Retrieval retry workflow

        if (
            workflow_decision
            == "retry_retrieval"
        ):

            return "retrieval"

        # Intelligent report refinement workflow

        if (
            workflow_decision
            == "report_refinement"
        ):

            return (
                "report_refinement"
            )

        # Conversational report chat workflow

        if (
            workflow_decision
            == "report_chat"
        ):

            return "report_chat"

        # Report generation workflow

        if (
            workflow_decision
            == "generate_report"
        ):

            return (
                "report_generation"
            )

        # Document generation workflow

        if (
            workflow_decision
            == "generate_document"
        ):

            return (
                "document_generation"
            )

        # Workflow completion

        if (
            workflow_decision
            == "complete"
        ):

            return END

        # Default workflow routing

        routing_map = {
            CurrentStep.START.value:
                "query_decomposition",

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
            current_step,
            END,
        )