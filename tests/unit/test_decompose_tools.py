import pytest
from unittest.mock import Mock, patch

from tools.decompose_tools import (
    DecompositionTools,
)
from state.constants import Domain
from state.models import (
    SubQuery,
    SubQueryList,
)


class TestDecompositionTools:
    """Unit tests for DecompositionTools."""

    @patch(
        "tools.decompose_tools.LLMFactory.create_qwen_llm"
    )
    def test_decompose_query_success(
        self,
        mock_create_llm,
    ):
        """Test successful query decomposition."""

        mock_llm = Mock()

        mock_structured_llm = Mock()

        mock_response = SubQueryList(
            sub_queries=[
                SubQuery(
                    query=(
                        "quantum computing "
                        "applications"
                    ),
                    domain=Domain.WEB,
                    priority=1,
                ),
                SubQuery(
                    query=(
                        "quantum computing "
                        "history"
                    ),
                    domain=(
                        Domain.WIKIPEDIA
                    ),
                    priority=2,
                ),
            ]
        )

        mock_chain = Mock()

        mock_chain.invoke.return_value = (
            mock_response
        )

        mock_prompt = Mock()

        mock_prompt.__or__ = Mock(
            return_value=mock_chain
        )

        mock_llm.with_structured_output.return_value = (
            mock_structured_llm
        )

        mock_create_llm.return_value = (
            mock_llm
        )

        with patch(
            "tools.decompose_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            result = (
                DecompositionTools.decompose_query(
                    query=(
                        "Quantum computing "
                        "applications"
                    )
                )
            )

        assert len(result) == 2

        assert all(
            isinstance(
                sub_query,
                SubQuery,
            )
            for sub_query in result
        )

    def test_validate_sub_queries_success(
        self,
    ):
        """Test valid sub-query validation."""

        sub_queries = [
            SubQuery(
                query="AI tutorials",
                domain=Domain.WEB,
                priority=1,
            ),
            SubQuery(
                query="AI history",
                domain=(
                    Domain.WIKIPEDIA
                ),
                priority=2,
            ),
        ]

        result = (
            DecompositionTools
            .validate_sub_queries(
                sub_queries
            )
        )

        assert result is True

    def test_validate_sub_queries_empty(
        self,
    ):
        """Test validation with empty sub-queries."""

        result = (
            DecompositionTools
            .validate_sub_queries(
                []
            )
        )

        assert result is False

    def test_validate_sub_queries_exceeds_limit(
        self,
    ):
        """Test validation when max limit exceeded."""

        sub_queries = [
            SubQuery(
                query=f"Query {index}",
                domain=(
                    Domain.WEB
                    if index % 2 == 0
                    else Domain.WIKIPEDIA
                ),
                priority=5,
            )
            for index in range(6)
        ]

        result = (
            DecompositionTools
            .validate_sub_queries(
                sub_queries
            )
        )

        assert result is False

    def test_validate_sub_queries_single_domain(
        self,
    ):
        """Test validation failure with one domain only."""

        sub_queries = [
            SubQuery(
                query="AI tutorials",
                domain=Domain.WEB,
                priority=1,
            ),
            SubQuery(
                query="AI tools",
                domain=Domain.WEB,
                priority=2,
            ),
        ]

        result = (
            DecompositionTools
            .validate_sub_queries(
                sub_queries
            )
        )

        assert result is False

    def test_validate_sub_queries_with_arxiv(
        self,
    ):
        """Test validation including arxiv domain."""

        sub_queries = [
            SubQuery(
                query="LLM research papers",
                domain=Domain.ARXIV,
                priority=1,
            ),
            SubQuery(
                query="LLM applications",
                domain=Domain.WEB,
                priority=2,
            ),
        ]

        result = (
            DecompositionTools
            .validate_sub_queries(
                sub_queries
            )
        )

        assert result is True