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
    def store_documents(
        documents: list[Document],
    ) -> None:
        vector_store = (
            VectorStoreManager.get_vector_store()
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
    def semantic_search(
        query: str,
        top_k: int = 5,
    ) -> list[Document]:
        vector_store = (
            VectorStoreManager.get_vector_store()
        )

        results = (
            vector_store.similarity_search(
                query=query,
                k=top_k,
            )
        )

        documents: list[Document] = []

        for result in results:
            documents.append(
                Document(
                    source=Domain(
                        result.metadata["source"]
                    ),
                    title=result.metadata[
                        "title"
                    ],
                    content=(
                        result.page_content
                    ),
                    url=result.metadata.get(
                        "url",
                    ),
                )
            )

        return documents