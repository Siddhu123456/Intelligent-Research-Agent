from langchain_chroma import Chroma

from utils.embedding_factory import (
    EmbeddingFactory,
)


class VectorStoreManager:
    """Manages ChromaDB vector store."""

    _vector_store: Chroma | None = None

    @classmethod
    def get_vector_store(
        cls,
    ) -> Chroma:
        if cls._vector_store is None:
            embeddings = (
                EmbeddingFactory.get_embeddings()
            )

            cls._vector_store = Chroma(
                collection_name=(
                    "research_documents"
                ),
                embedding_function=embeddings,
                persist_directory=(
                    "./chroma_db"
                ),
            )

        return cls._vector_store