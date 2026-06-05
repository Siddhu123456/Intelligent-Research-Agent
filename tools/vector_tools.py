from langchain_core.documents import (
    Document as LangchainDocument,
)

from state.models import Document
from state.constants import Domain
from utils.vector_store import (
    VectorStoreManager,
)


class VectorTools:
    """Tools for vector database operations."""

    @staticmethod
    @staticmethod
    def store_documents(
        documents: list[Document],
        session_id: str,
    ) -> None:
        vector_store = (
            VectorStoreManager.get_vector_store(
                session_id=session_id,
            )
        )

        langchain_documents = []

        ids = []

        for document in documents:
            langchain_documents.append(
                LangchainDocument(
                    page_content=(
                        document.content
                    ),
                    metadata={
                        "title":
                            document.title,
                        "source":
                            document.source.value,
                        "url":
                            document.url,
                    },
                )
            )

            ids.append(document.id)

        vector_store.add_documents(
            documents=langchain_documents,
            ids=ids,
        )

    @staticmethod
    @staticmethod
    def semantic_search(
        query: str,
        session_id: str,
        top_k: int = 5,
    ) -> list[Document]:
        vector_store = (
            VectorStoreManager.get_vector_store(
                session_id=session_id,
            )
        )

        results = (
            vector_store.similarity_search(
                query=query,
                k=top_k,
            )
        )

        documents: list[Document] = []

        for result in results:
            # Safely extract metadata values with sensible defaults
            src_val = result.metadata.get("source", "vector_db") if hasattr(result, "metadata") else "vector_db"
            try:
                src = Domain(src_val)
            except Exception:
                src = Domain.VECTOR_DB

            title = (result.metadata.get("title") if hasattr(result, "metadata") else None) or ""
            url = (result.metadata.get("url") if hasattr(result, "metadata") else None) or None
            content = result.page_content if getattr(result, "page_content", None) is not None else ""

            documents.append(
                Document(
                    source=src,
                    title=title,
                    content=(
                        content
                    ),
                    url=url,
                )
            )

        return documents