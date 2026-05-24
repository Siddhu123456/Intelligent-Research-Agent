from copy import (
    deepcopy,
)

from state.constants import (
    CurrentStep,
)

from state.schema import (
    ResearchState,
)


class StateManager:
    """Centralized workflow state lifecycle manager."""

    PERSISTENT_FIELDS = [
        "session_id",
        "messages",
        "conversation_turn",
        "conversation_summary",
        "report_version_history",
        "active_report",
        "report_title",
        "compressed_report_context",
        "report_sections",
        "report_section_order",
        "generated_pdf",
        "mode",
    ]

    WORKFLOW_DEFAULTS = {
        "contextualized_query": "",
        "sub_queries": [],
        "retrieved_documents": [],
        "key_findings": [],
        "analysis_summary": "",
        "citations": [],
        "workflow_decision": "continue",
        "retry_count": 0,
        "low_confidence": False,
        "error": None,
        "refinement_query": "",
        "report_chat_query": "",
        "report_chat_response": "",
        "report_sections": {},
        "report_section_order": [],
        "generated_pdf": None,
        "current_step": (
            CurrentStep.START.value
        ),
    }

    @staticmethod
    def preserve_persistent_state(
        state: ResearchState,
    ) -> dict:
        """Extract persistent workspace state."""

        persistent_state = {}

        for field in (
            StateManager
            .PERSISTENT_FIELDS
        ):

            persistent_state[field] = (
                deepcopy(
                    state.get(
                        field,
                    )
                )
            )

        return persistent_state

    @staticmethod
    def reset_workflow_state(
        state: ResearchState,
    ) -> ResearchState:
        """Reset transient workflow state."""

        for (
            field,
            value,
        ) in (
            StateManager
            .WORKFLOW_DEFAULTS
            .items()
        ):

            state[field] = (
                deepcopy(value)
            )

        return state

    @staticmethod
    def prepare_next_workflow(
        previous_state: ResearchState,
        query: str,
        mode: str = (
            "REPORT_GENERATION"
        ),
        refinement_query: str = "",
        report_chat_query: str = "",
    ) -> ResearchState:
        """Prepare state for next workflow."""

        state = deepcopy(
            previous_state,
        )

        persistent_state = (
            StateManager
            .preserve_persistent_state(
                state,
            )
        )

        state.clear()

        # Restore persistent state

        state.update(
            persistent_state,
        )

        # Reset transient workflow state

        state.update(
            deepcopy(
                StateManager
                .WORKFLOW_DEFAULTS
            )
        )

        # Re-apply persistent state so defaults do not overwrite preserved values
        state.update(
            persistent_state,
        )

        # Preserve report workspace
        # for conversational workflows

        if mode in [
            "REPORT_CHAT",
            "REPORT_REFINEMENT",
            "DOCUMENT_GENERATION",
        ]:

            state["report_sections"] = (
                persistent_state.get(
                    "report_sections",
                    {},
                )
            )

            state["report_section_order"] = (
                persistent_state.get(
                    "report_section_order",
                    [],
                )
            )

            state["active_report"] = (
                persistent_state.get(
                    "active_report",
                    "",
                )
            )

            state[
                "compressed_report_context"
            ] = (
                persistent_state.get(
                    "compressed_report_context",
                    "",
                )
            )

        # Set workflow-specific state

        state["query"] = query

        state["mode"] = mode

        state[
            "refinement_query"
        ] = refinement_query

        state[
            "report_chat_query"
        ] = report_chat_query

        # Reset generated PDF
        # for non-document workflows

        if (
            mode
            != "DOCUMENT_GENERATION"
        ):

            state[
                "generated_pdf"
            ] = None

        return state