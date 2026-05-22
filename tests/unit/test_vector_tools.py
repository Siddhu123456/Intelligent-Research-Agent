import pytest
from unittest.mock import Mock, patch

from langchain_core.documents import (
    Document as LangchainDocument,
)

from tools.vector_tools import (
    VectorTools,
)
from state.models import Document
from state.constants import Domain


class TestVectorTools:
    """Unit tests for VectorTools."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents."""

        return [
            Document(
                source=Domain.WEB,
                title="AI Research",
                content=(
                    "Artificial Intelligence "
                    "improves automation."
                ),
                url="https://example.com/ai",
            ),
            Document(
                source=Domain.ARXIV,
                title="Quantum Computing",
                content=(
                    "Quantum computing "
                    "research content."
                ),
                url="https://arxiv.org/1234",
            ),
        ]

    @patch(
        "tools.vector_tools.VectorStoreManager.get_vector_store"
    )
    def test_store_documents_success(
        self,
        mock_get_vector_store,
        sample_documents,
    ):
        """Test successful document storage."""

        mock_vector_store = Mock()

        mock_get_vector_store.return_value = (
            mock_vector_store
        )

        VectorTools.store_documents(
            documents=sample_documents,
            session_id="session_123",
        )

        mock_vector_store.add_documents.assert_called_once()

        call_args = (
            mock_vector_store
            .add_documents
            .call_args
        )

        stored_documents = (
            call_args.kwargs[
                "documents"
            ]
        )

        stored_ids = (
            call_args.kwargs[
                "ids"
            ]
        )

        assert len(stored_documents) == 2

        assert len(stored_ids) == 2

        assert all(
            isinstance(
                document,
                LangchainDocument,
            )
            for document in stored_documents
        )

    @patch(
        "tools.vector_tools.VectorStoreManager.get_vector_store"
    )
    def test_store_documents_metadata(
        self,
        mock_get_vector_store,
        sample_documents,
    ):
        """Test stored metadata correctness."""

        mock_vector_store = Mock()

        mock_get_vector_store.return_value = (
            mock_vector_store
        )

        VectorTools.store_documents(
            documents=sample_documents,
            session_id="session_456",
        )

        call_args = (
            mock_vector_store
            .add_documents
            .call_args
        )

        stored_documents = (
            call_args.kwargs[
                "documents"
            ]
        )

        first_document = (
            stored_documents[0]
        )

        assert (
            first_document.metadata[
                "title"
            ]
            == "AI Research"
        )

        assert (
            first_document.metadata[
                "source"
            ]
            == "web"
        )

        assert (
            first_document.metadata[
                "url"
            ]
            == "https://example.com/ai"
        )

    @patch(
        "tools.vector_tools.VectorStoreManager.get_vector_store"
    )
    def test_store_documents_empty(
        self,
        mock_get_vector_store,
    ):
        """Test storing empty document list."""

        mock_vector_store = Mock()

        mock_get_vector_store.return_value = (
            mock_vector_store
        )

        VectorTools.store_documents(
            documents=[],
            session_id="session_empty",
        )

        mock_vector_store.add_documents.assert_called_once_with(
            documents=[],
            ids=[],
        )

    @patch(
        "tools.vector_tools.VectorStoreManager.get_vector_store"
    )
    def test_semantic_search_success(
        self,
        mock_get_vector_store,
    ):
        """Test successful semantic search."""

        mock_vector_store = Mock()

        mock_result_1 = Mock()
        mock_result_1.page_content = (
            "AI automation content."
        )

        mock_result_1.metadata = {
            "title":
            "AI Research",
            "source":
            "web",
            "url":
            "https://example.com/ai",
        }

        mock_result_2 = Mock()
        mock_result_2.page_content = (
            "Quantum computing content."
        )

        mock_result_2.metadata = {
            "title":
            "Quantum Computing",
            "source":
            "arxiv",
            "url":
            "https://arxiv.org/1234",
        }

        mock_vector_store.similarity_search.return_value = [
            mock_result_1,
            mock_result_2,
        ]

        mock_get_vector_store.return_value = (
            mock_vector_store
        )

        result = (
            VectorTools.semantic_search(
                query=(
                    "Artificial Intelligence"
                ),
                session_id="session_789",
                top_k=2,
            )
        )

        assert len(result) == 2

        assert all(
            isinstance(
                document,
                Document,
            )
            for document in result
        )

        assert (
            result[0].title
            == "AI Research"
        )

        assert (
            result[1].source
            == Domain.ARXIV
        )

    @patch(
        "tools.vector_tools.VectorStoreManager.get_vector_store"
    )
    def test_semantic_search_calls_similarity_search(
        self,
        mock_get_vector_store,
    ):
        """Test similarity search invocation."""

        mock_vector_store = Mock()

        mock_vector_store.similarity_search.return_value = []

        mock_get_vector_store.return_value = (
            mock_vector_store
        )

        VectorTools.semantic_search(
            query="Machine Learning",
            session_id="session_test",
            top_k=3,
        )

        mock_vector_store.similarity_search.assert_called_once_with(
            query="Machine Learning",
            k=3,
        )

    @patch(
        "tools.vector_tools.VectorStoreManager.get_vector_store"
    )
    def test_semantic_search_empty_results(
        self,
        mock_get_vector_store,
    ):
        """Test semantic search with no results."""

        mock_vector_store = Mock()

        mock_vector_store.similarity_search.return_value = []

        mock_get_vector_store.return_value = (
            mock_vector_store
        )

        result = (
            VectorTools.semantic_search(
                query="Unknown Topic",
                session_id="session_none",
            )
        )

        assert result == []