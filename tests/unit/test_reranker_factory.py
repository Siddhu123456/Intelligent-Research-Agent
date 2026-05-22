from unittest.mock import patch

from utils.reranker_factory import (
    RerankerFactory,
)


class TestRerankerFactory:
    """Unit tests for RerankerFactory."""

    def setup_method(
        self,
    ):
        """Reset singleton before each test."""

        RerankerFactory._reranker = (
            None
        )

    @patch(
        "utils.reranker_factory.CrossEncoder"
    )
    def test_get_reranker_creates_model(
        self,
        mock_cross_encoder,
    ):
        """Test reranker creation."""

        mock_model = (
            mock_cross_encoder
            .return_value
        )

        result = (
            RerankerFactory
            .get_reranker()
        )

        mock_cross_encoder.assert_called_once_with(
            (
                "cross-encoder/"
                "ms-marco-MiniLM-L-6-v2"
            ),
        )

        assert (
            result
            == mock_model
        )

    @patch(
        "utils.reranker_factory.CrossEncoder"
    )
    def test_get_reranker_singleton_behavior(
        self,
        mock_cross_encoder,
    ):
        """Test singleton reranker reuse."""

        first_model = (
            RerankerFactory
            .get_reranker()
        )

        second_model = (
            RerankerFactory
            .get_reranker()
        )

        assert (
            first_model
            == second_model
        )

        mock_cross_encoder.assert_called_once()

    @patch(
        "utils.reranker_factory.CrossEncoder"
    )
    def test_get_reranker_returns_existing_model(
        self,
        mock_cross_encoder,
    ):
        """Test returning cached reranker."""

        mock_existing_model = (
            object()
        )

        RerankerFactory._reranker = (
            mock_existing_model
        )

        result = (
            RerankerFactory
            .get_reranker()
        )

        assert (
            result
            == mock_existing_model
        )

        mock_cross_encoder.assert_not_called()

    @patch(
        "utils.reranker_factory.CrossEncoder"
    )
    def test_get_reranker_model_name(
        self,
        mock_cross_encoder,
    ):
        """Test reranker model configuration."""

        RerankerFactory.get_reranker()

        args, _ = (
            mock_cross_encoder
            .call_args
        )

        assert (
            args[0]
            == (
                "cross-encoder/"
                "ms-marco-MiniLM-L-6-v2"
            )
        )