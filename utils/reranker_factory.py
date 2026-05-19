from sentence_transformers import (
    CrossEncoder,
)


class RerankerFactory:
    """Factory for reranking models."""

    _reranker: CrossEncoder | None = None

    @classmethod
    def get_reranker(
        cls,
    ) -> CrossEncoder:
        if cls._reranker is None:
            cls._reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            )

        return cls._reranker