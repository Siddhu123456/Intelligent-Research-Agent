from pathlib import Path
from unittest.mock import Mock, patch

from utils.vector_store import (
    VectorStoreManager,
)


class TestVectorStoreManager:
    """Unit tests for VectorStoreManager."""

    def setup_method(
        self,
    ):
        """Reset cached vector stores."""

        VectorStoreManager._vector_stores = {}

    @patch(
        "utils.vector_store.Chroma"
    )
    @patch(
        "utils.vector_store.EmbeddingFactory.get_embeddings"
    )
    def test_get_vector_store_creates_store(
        self,
        mock_get_embeddings,
        mock_chroma,
    ):
        """Test vector store creation."""

        mock_embeddings = Mock()

        mock_get_embeddings.return_value = (
            mock_embeddings
        )

        mock_store = (
            mock_chroma.return_value
        )

        result = (
            VectorStoreManager
            .get_vector_store(
                session_id="session_1"
            )
        )

        mock_chroma.assert_called_once_with(
            collection_name=(
                "research_session_1"
            ),
            embedding_function=(
                mock_embeddings
            ),
            persist_directory=(
                "./chroma_db/"
                "research_session_1"
            ),
        )

        assert (
            result
            == mock_store
        )

    @patch(
        "utils.vector_store.Chroma"
    )
    @patch(
        "utils.vector_store.EmbeddingFactory.get_embeddings"
    )
    def test_get_vector_store_singleton_behavior(
        self,
        mock_get_embeddings,
        mock_chroma,
    ):
        """Test cached vector store reuse."""

        first_store = (
            VectorStoreManager
            .get_vector_store(
                session_id="session_2"
            )
        )

        second_store = (
            VectorStoreManager
            .get_vector_store(
                session_id="session_2"
            )
        )

        assert (
            first_store
            == second_store
        )

        mock_chroma.assert_called_once()

    @patch(
        "utils.vector_store.Chroma"
    )
    @patch(
        "utils.vector_store.EmbeddingFactory.get_embeddings"
    )
    def test_get_vector_store_different_sessions(
        self,
        mock_get_embeddings,
        mock_chroma,
    ):
        """Test separate stores for sessions."""

        mock_store_one = Mock()

        mock_store_two = Mock()

        mock_chroma.side_effect = [
            mock_store_one,
            mock_store_two,
        ]

        store_one = (
            VectorStoreManager
            .get_vector_store(
                session_id="session_a"
            )
        )

        store_two = (
            VectorStoreManager
            .get_vector_store(
                session_id="session_b"
            )
        )

        assert (
            store_one
            != store_two
        )

        assert (
            mock_chroma.call_count
            == 2
        )

    def test_clear_persisted_data_removes_cached_store(
        self,
    ):
        """Test cached vector store removal."""

        VectorStoreManager._vector_stores[
            "research_session_3"
        ] = Mock()

        VectorStoreManager.clear_persisted_data(
            session_id="session_3"
        )

        assert (
            "research_session_3"
            not in
            VectorStoreManager._vector_stores
        )

    @patch(
        "utils.vector_store.shutil.rmtree"
    )
    @patch(
        "utils.vector_store.Path.exists"
    )
    def test_clear_persisted_data_removes_directory(
        self,
        mock_exists,
        mock_rmtree,
    ):
        """Test persisted directory deletion."""

        mock_exists.return_value = (
            True
        )

        VectorStoreManager.clear_persisted_data(
            session_id="session_4"
        )

        mock_rmtree.assert_called_once()

        called_path = (
            mock_rmtree.call_args[0][0]
        )

        assert isinstance(
            called_path,
            Path,
        )

    @patch(
        "utils.vector_store.shutil.rmtree"
    )
    @patch(
        "utils.vector_store.Path.exists"
    )
    def test_clear_persisted_data_missing_directory(
        self,
        mock_exists,
        mock_rmtree,
    ):
        """Test cleanup with missing directory."""

        mock_exists.return_value = (
            False
        )

        VectorStoreManager.clear_persisted_data(
            session_id="session_5"
        )

        mock_rmtree.assert_not_called()

    @patch(
        "utils.vector_store.shutil.rmtree"
    )
    @patch(
        "utils.vector_store.Path.exists"
    )
    def test_clear_persisted_data_handles_exception(
        self,
        mock_exists,
        mock_rmtree,
    ):
        """Test cleanup exception handling."""

        mock_exists.return_value = (
            True
        )

        mock_rmtree.side_effect = (
            Exception(
                "Permission denied"
            )
        )

        VectorStoreManager.clear_persisted_data(
            session_id="session_6"
        )

        # Test passes if no exception raised

        assert True