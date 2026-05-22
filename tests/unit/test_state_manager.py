from copy import deepcopy

from state.constants import (
    CurrentStep,
)
from utils.state_manager import (
    StateManager,
)


class TestStateManager:
    """Unit tests for StateManager."""

    def test_preserve_persistent_state(
        self,
    ):
        """Test persistent state extraction."""

        state = {
            "session_id":
            "session_1",
            "messages":
            ["hello"],
            "conversation_turn":
            2,
            "active_report":
            "report",
            "query":
            "temporary query",
            "retry_count":
            5,
        }

        result = (
            StateManager
            .preserve_persistent_state(
                state
            )
        )

        assert (
            result["session_id"]
            == "session_1"
        )

        assert (
            result["messages"]
            == ["hello"]
        )

        assert (
            "query"
            not in result
        )

        assert (
            "retry_count"
            not in result
        )

    def test_preserve_persistent_state_deepcopy(
        self,
    ):
        """Test persistent state deep copy."""

        state = {
            "messages":
            ["msg1"],
            "session_id":
            "session_1",
        }

        result = (
            StateManager
            .preserve_persistent_state(
                state
            )
        )

        result["messages"].append(
            "msg2"
        )

        assert (
            state["messages"]
            == ["msg1"]
        )

    def test_reset_workflow_state(
        self,
    ):
        """Test workflow reset."""

        state = {
            "query":
            "AI",
            "retry_count":
            3,
            "low_confidence":
            True,
        }

        result = (
            StateManager
            .reset_workflow_state(
                state
            )
        )

        assert (
            result[
                "contextualized_query"
            ]
            == ""
        )

        assert (
            result[
                "retry_count"
            ]
            == 0
        )

        assert (
            result[
                "low_confidence"
            ]
            is False
        )

        assert (
            result[
                "current_step"
            ]
            == CurrentStep.START.value
        )

    def test_reset_workflow_state_returns_same_object(
        self,
    ):
        """Test reset modifies same object."""

        state = {}

        result = (
            StateManager
            .reset_workflow_state(
                state
            )
        )

        assert (
            result
            is state
        )

    def test_prepare_next_workflow_basic(
        self,
    ):
        """Test preparing next workflow."""

        previous_state = {
            "session_id":
            "session_123",
            "messages":
            ["hello"],
            "query":
            "old query",
            "retry_count":
            4,
            "low_confidence":
            True,
        }

        result = (
            StateManager
            .prepare_next_workflow(
                previous_state=
                previous_state,
                query=(
                    "new query"
                ),
            )
        )

        assert (
            result["query"]
            == "new query"
        )

        assert (
            result["mode"]
            == "REPORT_GENERATION"
        )

        assert (
            result["retry_count"]
            == 0
        )

        assert (
            result["low_confidence"]
            is False
        )

        assert (
            result["session_id"]
            == "session_123"
        )

    def test_prepare_next_workflow_custom_mode(
        self,
    ):
        """Test workflow with custom mode."""

        previous_state = {}

        result = (
            StateManager
            .prepare_next_workflow(
                previous_state=
                previous_state,
                query="AI",
                mode=(
                    "DOCUMENT_GENERATION"
                ),
            )
        )

        assert (
            result["mode"]
            == (
                "DOCUMENT_GENERATION"
            )
        )

    def test_prepare_next_workflow_refinement_query(
        self,
    ):
        """Test refinement query assignment."""

        result = (
            StateManager
            .prepare_next_workflow(
                previous_state={},
                query="AI",
                refinement_query=(
                    "Improve analysis"
                ),
            )
        )

        assert (
            result[
                "refinement_query"
            ]
            == (
                "Improve analysis"
            )
        )

    def test_prepare_next_workflow_report_chat_query(
        self,
    ):
        """Test report chat query assignment."""

        result = (
            StateManager
            .prepare_next_workflow(
                previous_state={},
                query="AI",
                report_chat_query=(
                    "What is AI?"
                ),
            )
        )

        assert (
            result[
                "report_chat_query"
            ]
            == "What is AI?"
        )

    def test_prepare_next_workflow_resets_generated_pdf(
        self,
    ):
        """Test PDF reset for non-document workflows."""

        previous_state = {
            "generated_pdf":
            "old.pdf"
        }

        result = (
            StateManager
            .prepare_next_workflow(
                previous_state=
                previous_state,
                query="AI",
                mode=(
                    "REPORT_GENERATION"
                ),
            )
        )

        assert (
            result[
                "generated_pdf"
            ]
            is None
        )

    def test_prepare_next_workflow_preserves_pdf_document_mode(
        self,
    ):
        """Test PDF preservation in document mode."""

        previous_state = {
            "generated_pdf":
            "report.pdf"
        }

        result = (
            StateManager
            .prepare_next_workflow(
                previous_state=
                previous_state,
                query="AI",
                mode=(
                    "DOCUMENT_GENERATION"
                ),
            )
        )

        assert (
            result[
                "generated_pdf"
            ]
            == "report.pdf"
        )

    def test_prepare_next_workflow_deepcopy(
        self,
    ):
        """Test workflow state deep copy isolation."""

        previous_state = {
            "messages":
            ["hello"],
        }

        result = (
            StateManager
            .prepare_next_workflow(
                previous_state=
                previous_state,
                query="AI",
            )
        )

        result["messages"].append(
            "new"
        )

        assert (
            previous_state[
                "messages"
            ]
            == ["hello"]
        )