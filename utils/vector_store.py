from langchain_chroma import Chroma

from utils.embedding_factory import (
    EmbeddingFactory,
)

from pathlib import Path
import shutil


class VectorStoreManager:
    """Manages ChromaDB vector store and persisted data.

    Provides a helper to clear persisted Chroma files (useful
    for per-session cleanup workflows).
    """

    _vector_stores: dict[
        str,
        Chroma,
    ] = {}

    @classmethod
    def get_vector_store(
        cls,
        session_id: str,
    ) -> Chroma:

        collection_name = (
            f"research_{session_id}"
        )

        if (
            collection_name
            not in cls._vector_stores
        ):

            embeddings = (
                EmbeddingFactory
                .get_embeddings()
            )

            persist_directory = (
                f"./chroma_db/"
                f"{collection_name}"
            )

            cls._vector_stores[
                collection_name
            ] = Chroma(
                collection_name=(
                    collection_name
                ),
                embedding_function=(
                    embeddings
                ),
                persist_directory=(
                    persist_directory
                ),
            )

        return cls._vector_stores[
            collection_name
        ]
    
    @classmethod
    def clear_persisted_data(
        cls,
        session_id: str,
    ) -> None:
        """
        Remove session-specific
        Chroma collection data.
        """

        collection_name = (
            f"research_{session_id}"
        )

        # Remove cached vectorstore

        if (
            collection_name
            in cls._vector_stores
        ):

            del cls._vector_stores[
                collection_name
            ]

        chroma_dir = Path(
            f"./chroma_db/"
            f"{collection_name}"
        )

        if chroma_dir.exists():

            try:

                shutil.rmtree(
                    chroma_dir
                )

            except Exception:

                pass