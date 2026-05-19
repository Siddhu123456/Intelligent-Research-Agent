from state.models import Document
from utils.reranker_factory import (
    RerankerFactory,
)


class RerankingTools:
    """Tools for semantic reranking."""

    @staticmethod
    def rerank_documents(
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        if not documents:
            return []

        reranker = (
            RerankerFactory.get_reranker()
        )

        pairs = [
            (
                query,
                document.content,
            )
            for document in documents
        ]

        scores = reranker.predict(
            pairs,
        )

        scored_documents = list(
            zip(
                documents,
                scores,
            )
        )

        scored_documents.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        reranked_documents = [
            document
            for document, _ in scored_documents[
                :top_k
            ]
        ]

        return reranked_documents