import pytest

from tools.supervisor_tools import (
    SupervisorTools,
)


class TestSupervisorTools:
    """Unit tests for SupervisorTools."""

    @pytest.fixture
    def base_state(self):
        """Create base research state."""

        return {
            "low_confidence": False,
            "retry_count": 0,
            "max_retries": 3,
            "retrieved_documents": [],
        }

    def test_should_retry_retrieval_true(
        self,
        base_state,
    ):
        """Test retry decision returns True."""

        base_state[
            "low_confidence"
        ] = True

        base_state[
            "retry_count"
        ] = 1

        result = (
            SupervisorTools
            .should_retry_retrieval(
                base_state
            )
        )

        assert result is True

    def test_should_retry_retrieval_false_no_confidence(
        self,
        base_state,
    ):
        """Test retry decision when confidence is sufficient."""

        base_state[
            "low_confidence"
        ] = False

        result = (
            SupervisorTools
            .should_retry_retrieval(
                base_state
            )
        )

        assert result is False

    def test_should_retry_retrieval_false_max_retries(
        self,
        base_state,
    ):
        """Test retry decision when retries exceeded."""

        base_state[
            "low_confidence"
        ] = True

        base_state[
            "retry_count"
        ] = 3

        result = (
            SupervisorTools
            .should_retry_retrieval(
                base_state
            )
        )

        assert result is False

    def test_should_refine_query_true(
        self,
        base_state,
    ):
        """Test query refinement required."""

        base_state[
            "retrieved_documents"
        ] = [
            "doc1",
            "doc2",
        ]

        result = (
            SupervisorTools
            .should_refine_query(
                base_state
            )
        )

        assert result is True

    def test_should_refine_query_false(
        self,
        base_state,
    ):
        """Test query refinement not required."""

        base_state[
            "retrieved_documents"
        ] = [
            "doc1",
            "doc2",
            "doc3",
        ]

        result = (
            SupervisorTools
            .should_refine_query(
                base_state
            )
        )

        assert result is False

    def test_increment_retry(
        self,
        base_state,
    ):
        """Test retry counter increment."""

        assert (
            base_state[
                "retry_count"
            ]
            == 0
        )

        SupervisorTools.increment_retry(
            base_state
        )

        assert (
            base_state[
                "retry_count"
            ]
            == 1
        )

    def test_increment_retry_multiple_times(
        self,
        base_state,
    ):
        """Test multiple retry increments."""

        SupervisorTools.increment_retry(
            base_state
        )

        SupervisorTools.increment_retry(
            base_state
        )

        SupervisorTools.increment_retry(
            base_state
        )

        assert (
            base_state[
                "retry_count"
            ]
            == 3
        )