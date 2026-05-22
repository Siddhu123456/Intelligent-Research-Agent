from state.constants import (
    CurrentStep,
)

from utils.state_reset import (
    StateResetUtility,
)


class TestStateResetUtility:
    """Unit tests for StateResetUtility."""

    def test_reset_workflow_state(
        self,
    ):
        """Test workflow state reset."""

        state = {
            "retrieved_documents":
            ["doc1", "doc2"],
            "key_findings":
            ["finding1"],
            "analysis_summary":
            "Old summary",
            "citations":
            ["citation1"],
            "retry_count":
            5,
            "low_confidence":
            True,
            "error":
            "Some error",
            "workflow_decision":
            "retry",
            "current_step":
            "PROCESSING",
        }

        result = (
            StateResetUtility
            .reset_workflow_state(
                state
            )
        )

        assert (
            result[
                "retrieved_documents"
            ]
            == []
        )

        assert (
            result[
                "key_findings"
            ]
            == []
        )

        assert (
            result[
                "analysis_summary"
            ]
            == ""
        )

        assert (
            result[
                "citations"
            ]
            == []
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
                "error"
            ]
            is None
        )

        assert (
            result[
                "workflow_decision"
            ]
            == "continue"
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
        """Test same state object is returned."""

        state = {}

        result = (
            StateResetUtility
            .reset_workflow_state(
                state
            )
        )

        assert (
            result
            is state
        )

    def test_reset_workflow_state_preserves_other_fields(
        self,
    ):
        """Test unrelated fields remain unchanged."""

        state = {
            "session_id":
            "session_123",
            "query":
            "AI research",
            "retrieved_documents":
            ["doc1"],
            "retry_count":
            2,
        }

        result = (
            StateResetUtility
            .reset_workflow_state(
                state
            )
        )

        assert (
            result["session_id"]
            == "session_123"
        )

        assert (
            result["query"]
            == "AI research"
        )

        assert (
            result[
                "retrieved_documents"
            ]
            == []
        )

        assert (
            result[
                "retry_count"
            ]
            == 0
        )

    def test_reset_workflow_state_multiple_calls(
        self,
    ):
        """Test repeated reset calls."""

        state = {
            "retrieved_documents":
            ["doc"],
            "retry_count":
            3,
            "low_confidence":
            True,
        }

        StateResetUtility.reset_workflow_state(
            state
        )

        StateResetUtility.reset_workflow_state(
            state
        )

        assert (
            state[
                "retrieved_documents"
            ]
            == []
        )

        assert (
            state[
                "retry_count"
            ]
            == 0
        )

        assert (
            state[
                "low_confidence"
            ]
            is False
        )