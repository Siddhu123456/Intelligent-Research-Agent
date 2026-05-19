from state.models import Document
from utils.llm_factory import (
    LLMFactory,
)


class CompressionTools:
    """Tools for contextual compression."""

    @staticmethod
    def compress_documents(
        query: str,
        documents: list[Document],
    ) -> list[Document]:
        llm = (
            LLMFactory.create_qwen_llm(
                temperature=0.1,
            )
        )

        compressed_documents = []

        for document in documents:
            prompt = f"""
You are a contextual compression agent.

User Query:
{query}

Document:
{document.content}

Extract ONLY information relevant
to the query.

Remove irrelevant details.
Keep factual information only.
"""

            response = llm.invoke(
                prompt,
            )

            compressed_documents.append(
                Document(
                    source=document.source,
                    title=document.title,
                    content=response.content,
                    url=document.url,
                )
            )

        return compressed_documents