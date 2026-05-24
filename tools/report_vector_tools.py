from uuid import uuid4

from langchain_core.documents import (
    Document as LangchainDocument,
)

from utils.reranker_factory import RerankerFactory
from utils.vector_store import (
    VectorStoreManager,
)


class ReportVectorTools:
    """Tools for report-level vector storage."""

    @staticmethod
    def clear_report_embeddings(
        session_id: str,
    ) -> None:
        """
        Remove old report embeddings
        for session.
        """

        vector_store = (
            VectorStoreManager
            .get_vector_store(
                session_id=session_id,
            )
        )

        collection = (
            vector_store._collection
        )

        results = collection.get(
            where={
                "$and": [
                    {
                        "chunk_type":
                            "report",
                    },
                    {
                        "session_id":
                            session_id,
                    },
                ]
            }
        )

        ids = results.get(
            "ids",
            [],
        )

        if ids:

            collection.delete(
                ids=ids,
            )

    @staticmethod
    def store_report(
        sections: dict[str, str],
        session_id: str,
    ) -> None:
        """
        Store report sections
        inside vector database.
        """

        if not sections:

            return

        vector_store = (
            VectorStoreManager
            .get_vector_store(
                session_id=session_id,
            )
        )

        documents = []

        ids = []

        for (
            section_name,
            content,
        ) in sections.items():

            if not isinstance(
                content,
                str,
            ):

                content = str(
                    content
                )

            if not content.strip():

                continue

            documents.append(
                LangchainDocument(
                    page_content=content,
                    metadata={

                        "chunk_type":
                            "report",

                        "section":
                            section_name,

                        "session_id":
                            session_id,
                    },
                )
            )

            ids.append(
                str(uuid4())
            )

        if documents:

            vector_store.add_documents(
                documents=documents,
                ids=ids,
            )

    @staticmethod
    def semantic_report_search(
        query: str,
        session_id: str,
        top_k: int = 5,
        retrieval_k: int = 15,
    ) -> list[dict]:
        """
        Perform semantic search
        with reranking over
        generated report sections.
        """

        vector_store = (
            VectorStoreManager
            .get_vector_store(
                session_id=session_id,
            )
        )

        # Initial broad semantic retrieval

        results = (
            vector_store.similarity_search(
                query=query,
                k=retrieval_k,
                filter={
                    "chunk_type":
                        "report",
                },
            )
        )

        if not results:

            return []

        # Convert vector results into internal documents

        documents = []

        for result in results:

            content = (
                result.page_content
            )

            if not isinstance(
                content,
                str,
            ):

                content = str(
                    content
                )

            if not content.strip():

                continue

            documents.append(
                {
                    "section":
                        result.metadata.get(
                            "section",
                            "",
                        ),

                    "content":
                        content,

                    "document":
                        result,
                }
            )

        if not documents:

            return []

        # Prepare reranker inputs

        reranker = (
            RerankerFactory
            .get_reranker()
        )

        pairs = [
            (
                query,
                document["content"],
            )
            for document in documents
        ]

        scores = reranker.predict(
            pairs,
        )

        # Combine scores with documents

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

        # Filter weak matches

        MIN_SCORE = 0.15

        reranked_results = []

        for (
            document,
            score,
        ) in scored_documents:

            if score < MIN_SCORE:

                continue

            reranked_results.append(
                {
                    "section":
                        document[
                            "section"
                        ],

                    "content":
                        document[
                            "content"
                        ],

                    "score":
                        float(score),
                }
            )

            if (
                len(
                    reranked_results
                )
                >= top_k
            ):

                break

        # Fallback if all scores filtered

        if not reranked_results:

            reranked_results = [
                {
                    "section":
                        document[
                            "section"
                        ],

                    "content":
                        document[
                            "content"
                        ],

                    "score":
                        float(score),
                }
                for (
                    document,
                    score,
                ) in scored_documents[
                    :top_k
                ]
            ]

        return reranked_results