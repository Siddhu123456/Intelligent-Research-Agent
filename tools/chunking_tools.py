from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from state.models import Document


class ChunkingTools:
    """Tools for intelligent document chunking."""

    _text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
    )

    @staticmethod
    def chunk_documents(
        documents: list[Document],
    ) -> list[Document]:
        chunked_documents: list[Document] = []

        for document in documents:
            chunks = (
                ChunkingTools
                ._text_splitter
                .split_text(
                    document.content,
                )
            )

            for index, chunk in enumerate(
                chunks,
            ):
                chunked_documents.append(
                    Document(
                        source=document.source,
                        title=(
                            f"{document.title} "
                            f"(Chunk {index + 1})"
                        ),
                        content=chunk,
                        url=document.url,
                    )
                )

        return chunked_documents