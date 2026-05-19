from uuid import uuid4

from state.constants import CurrentStep
from state.schema import ResearchState


class StateFactory:
    """Factory class for creating workflow states."""

    @staticmethod
    def create_initial_state(
        query: str,
    ) -> ResearchState:
        return {
            "query": query,
            "sub_queries": [],
            "retrieved_documents": [],
            "analysis_summary": "",
            "key_findings": [],
            "citations": [],
            "final_report": "",
            "current_step": (
                CurrentStep.START.value
            ),
            "low_confidence": False,
            "error": None,
            "messages": [],
            "run_metadata": {},
            "conversation_summary": "",
            "contextualized_query": query,
            "session_id": str(uuid4()),
            "conversation_turn": 1,
            "retry_count": 0,
            "max_retries": 2,
            "workflow_decision": "",
            "report_history": []
        }