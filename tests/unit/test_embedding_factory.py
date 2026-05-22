import pytest
from unittest.mock import patch

from utils.embedding_factory import (
    EmbeddingFactory,
)


class TestEmbeddingFactory:
    """Unit tests for EmbeddingFactory."""

    def setup_method(self):
        """Reset singleton before each test."""

        EmbeddingFactory._embedding_model = (
            None
        )

    @patch(
        "utils.embedding_factory.HuggingFaceEmbeddings"
    )
    def test_get_embeddings_creates_model(
        self,
        mock_huggingface_embeddings,
    ):
        """Test embedding model creation."""

        mock_model = (
            mock_huggingface_embeddings
            .return_value
        )

        result = (
            EmbeddingFactory
            .get_embeddings()
        )

        mock_huggingface_embeddings.assert_called_once_with(
            model_name=(
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            ),
            model_kwargs={
                "device": "cpu",
            },
            encode_kwargs={
                "normalize_embeddings":
                True,
            },
        )

        assert result == mock_model

    @patch(
        "utils.embedding_factory.HuggingFaceEmbeddings"
    )
    def test_get_embeddings_singleton_behavior(
        self,
        mock_huggingface_embeddings,
    ):
        """Test singleton embedding reuse."""

        first_model = (
            EmbeddingFactory
            .get_embeddings()
        )

        second_model = (
            EmbeddingFactory
            .get_embeddings()
        )

        assert (
            first_model
            == second_model
        )

        mock_huggingface_embeddings.assert_called_once()

    @patch(
        "utils.embedding_factory.HuggingFaceEmbeddings"
    )
    def test_get_embeddings_returns_existing_model(
        self,
        mock_huggingface_embeddings,
    ):
        """Test returning cached embedding model."""

        mock_existing_model = object()

        EmbeddingFactory._embedding_model = (
            mock_existing_model
        )

        result = (
            EmbeddingFactory
            .get_embeddings()
        )

        assert (
            result
            == mock_existing_model
        )

        mock_huggingface_embeddings.assert_not_called()

    @patch(
        "utils.embedding_factory.HuggingFaceEmbeddings"
    )
    def test_get_embeddings_model_configuration(
        self,
        mock_huggingface_embeddings,
    ):
        """Test embedding configuration values."""

        EmbeddingFactory.get_embeddings()

        _, kwargs = (
            mock_huggingface_embeddings
            .call_args
        )

        assert (
            kwargs["model_kwargs"][
                "device"
            ]
            == "cpu"
        )

        assert (
            kwargs["encode_kwargs"][
                "normalize_embeddings"
            ]
            is True
        )

        assert (
            kwargs["model_name"]
            == (
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            )
        )