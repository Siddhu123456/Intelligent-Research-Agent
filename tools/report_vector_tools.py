from uuid import uuid4

from langchain_core.documents import (
    Document as LangchainDocument,
)

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
        top_k: int = 4,
    ) -> list[dict]:
        """
        Perform semantic search
        over generated report.
        """

        vector_store = (
            VectorStoreManager
            .get_vector_store(
                session_id=session_id,
            )
        )

        results = (
            vector_store.similarity_search(
                query=query,
                k=top_k,
                filter={
                    "chunk_type":
                        "report",
                },
            )
        )

        formatted_results = []

        for result in results:

            formatted_results.append(
                {
                    "section":
                        result.metadata.get(
                            "section",
                            "",
                        ),

                    "content":
                        result.page_content,
                }
            )

        return formatted_results