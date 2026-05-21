from uuid import (
    uuid4,
)

from state.constants import (
    CurrentStep,
)

from state.schema import (
    ResearchState,
)


class StateFactory:
    """Factory class for creating workflow states."""

    DEFAULT_MAX_RETRIES = 2

    DEFAULT_MODE = (
        "REPORT_GENERATION"
    )

    @staticmethod
    def create_initial_state(
        query: str,
    ) -> ResearchState:
        """Create initial workflow state."""

        return {
            # Core query state

            "query": query,

            "contextualized_query": query,

            # Workflow execution state

            "current_step": (
                CurrentStep.START.value
            ),

            "workflow_decision": (
                "continue"
            ),

            "retry_count": 0,

            "max_retries": (
                StateFactory
                .DEFAULT_MAX_RETRIES
            ),

            "low_confidence": False,

            "error": None,

            # Retrieval state

            "sub_queries": [],

            "retrieved_documents": [],

            # Analysis state

            "analysis_summary": "",

            "key_findings": [],

            "citations": [],

            # Report workspace state

            "active_report": "",
            
            "report_sections": {},
            
            "report_section_order": [],

            "refinement_query": "",

            "report_chat_query": "",

            "report_chat_response": "",

            "report_history": [],

            "report_version_history": [],
            
            "compressed_report_context": "",

            # Document generation state

            "generated_pdf": None,

            # Conversation memory state

            "messages": [],

            "conversation_summary": "",

            "conversation_turn": 1,

            # Session state

            "session_id": str(
                uuid4()
            ),

            # Workspace mode

            "mode": (
                StateFactory
                .DEFAULT_MODE
            ),

            # Execution metadata

            "run_metadata": {},
        }