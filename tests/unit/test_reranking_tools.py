import pytest
from unittest.mock import Mock, patch

from tools.reranking_tools import (
    RerankingTools,
)
from state.models import Document


class TestRerankingTools:
    """Unit tests for RerankingTools."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents."""

        return [
            Document(
                source="arxiv",
                title="AI Research",
                content=(
                    "Artificial Intelligence "
                    "improves automation."
                ),
                url="https://example.com/ai",
            ),
            Document(
                source="web",
                title="ML Trends",
                content=(
                    "Machine Learning is "
                    "widely adopted."
                ),
                url="https://example.com/ml",
            ),
            Document(
                source="wikipedia",
                title="Deep Learning",
                content=(
                    "Deep Learning uses "
                    "neural networks."
                ),
                url="https://example.com/dl",
            ),
        ]

    def test_rerank_documents_empty(
        self,
    ):
        """Test reranking with empty documents."""

        result = (
            RerankingTools
            .rerank_documents(
                query="AI",
                documents=[],
            )
        )

        assert result == []

    @patch(
        "tools.reranking_tools.RerankerFactory.get_reranker"
    )
    def test_rerank_documents_success(
        self,
        mock_get_reranker,
        sample_documents,
    ):
        """Test successful reranking."""

        mock_reranker = Mock()

        mock_reranker.predict.return_value = [
            0.95,
            0.75,
            0.85,
        ]

        mock_get_reranker.return_value = (
            mock_reranker
        )

        result = (
            RerankingTools
            .rerank_documents(
                query=(
                    "Artificial Intelligence"
                ),
                documents=sample_documents,
                top_k=2,
            )
        )

        assert len(result) == 2

        assert (
            result[0].title
            == "AI Research"
        )

        assert (
            result[1].title
            == "Deep Learning"
        )

    @patch(
        "tools.reranking_tools.RerankerFactory.get_reranker"
    )
    def test_rerank_documents_top_k(
        self,
        mock_get_reranker,
        sample_documents,
    ):
        """Test top_k filtering."""

        mock_reranker = Mock()

        mock_reranker.predict.return_value = [
            0.4,
            0.9,
            0.7,
        ]

        mock_get_reranker.return_value = (
            mock_reranker
        )

        result = (
            RerankingTools
            .rerank_documents(
                query="ML",
                documents=sample_documents,
                top_k=1,
            )
        )

        assert len(result) == 1

        assert (
            result[0].title
            == "ML Trends"
        )

    @patch(
        "tools.reranking_tools.RerankerFactory.get_reranker"
    )
    def test_rerank_documents_preserves_documents(
        self,
        mock_get_reranker,
        sample_documents,
    ):
        """Test returned objects remain Document instances."""

        mock_reranker = Mock()

        mock_reranker.predict.return_value = [
            0.6,
            0.5,
            0.4,
        ]

        mock_get_reranker.return_value = (
            mock_reranker
        )

        result = (
            RerankingTools
            .rerank_documents(
                query="AI",
                documents=sample_documents,
            )
        )

        assert all(
            isinstance(
                document,
                Document,
            )
            for document in result
        )

    @patch(
        "tools.reranking_tools.RerankerFactory.get_reranker"
    )
    def test_rerank_documents_predict_called(
        self,
        mock_get_reranker,
        sample_documents,
    ):
        """Test reranker predict invocation."""

        mock_reranker = Mock()

        mock_reranker.predict.return_value = [
            0.1,
            0.2,
            0.3,
        ]

        mock_get_reranker.return_value = (
            mock_reranker
        )

        query = "Deep Learning"

        RerankingTools.rerank_documents(
            query=query,
            documents=sample_documents,
        )

        expected_pairs = [
            (
                query,
                document.content,
            )
            for document in sample_documents
        ]

        mock_reranker.predict.assert_called_once_with(
            expected_pairs,
        )

    @patch(
        "tools.reranking_tools.RerankerFactory.get_reranker"
    )
    def test_rerank_documents_top_k_exceeds_length(
        self,
        mock_get_reranker,
        sample_documents,
    ):
        """Test top_k greater than document count."""

        mock_reranker = Mock()

        mock_reranker.predict.return_value = [
            0.3,
            0.2,
            0.1,
        ]

        mock_get_reranker.return_value = (
            mock_reranker
        )

        result = (
            RerankingTools
            .rerank_documents(
                query="AI",
                documents=sample_documents,
                top_k=10,
            )
        )

        assert (
            len(result) == 3
        )