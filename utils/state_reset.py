from state.constants import (
    CurrentStep,
)

from state.schema import (
    ResearchState,
)


class StateResetUtility:
    """Utility for resetting transient workflow state."""

    @staticmethod
    def reset_workflow_state(
        state: ResearchState,
    ) -> ResearchState:

        state["retrieved_documents"] = []

        state["key_findings"] = []

        state["analysis_summary"] = ""

        state["citations"] = []

        state["retry_count"] = 0

        state["low_confidence"] = False

        state["error"] = None

        state["workflow_decision"] = (
            "continue"
        )

        state["current_step"] = (
            CurrentStep.START.value
        )

        return state