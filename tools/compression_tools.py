from langchain_core.prompts import (
    ChatPromptTemplate,
)

from state.models import (
    Document,
)

from utils.json_parser import (
    JSONParser,
)

from utils.llm_factory import (
    LLMFactory,
)

from tools.prompts.compression_tools_prompts import (
    COMPRESSION_SYSTEM_PROMPT,
)


class CompressionTools:
    """Tools for contextual compression."""

    MIN_CONTENT_LENGTH = 30

    MAX_DOCUMENT_LENGTH = 4000

    @staticmethod
    def compress_documents(
        query: str,
        documents: list[Document],
    ) -> list[Document]:
        """Compress retrieved documents."""

        llm = (
            LLMFactory
            .create_qwen_llm(
                temperature=0.1,
            )
        )

        compressed_documents = []

        for document in documents:

            prompt = (
                ChatPromptTemplate
                .from_messages(
                    [
                        (
                            "system",
                            COMPRESSION_SYSTEM_PROMPT,
                        ),
                        (
                            "human",
                            (
                                "User Query:\n"
                                "{query}\n\n"
                                "Document:\n"
                                "{document_content}"
                            ),
                        ),
                    ]
                )
            )

            chain = (
                prompt
                | llm
            )

            response = chain.invoke(
                {
                    "query": query,
                    "document_content": (
                        document.content[
                            :CompressionTools
                            .MAX_DOCUMENT_LENGTH
                        ]
                    ),
                }
            )

            parsed_response = (
                JSONParser.safe_extract(
                    content=(
                        response.content
                    ),
                    fallback={
                        "content": (
                            document.content
                        )
                    },
                )
            )

            compressed_content = (
                parsed_response.get(
                    "content",
                    "",
                )
                .strip()
            )

            if (
                len(
                    compressed_content,
                )
                < CompressionTools
                .MIN_CONTENT_LENGTH
            ):

                continue

            compressed_documents.append(
                Document(
                    source=(
                        document.source
                    ),
                    title=(
                        document.title
                    ),
                    content=(
                        compressed_content[
                            :CompressionTools
                            .MAX_DOCUMENT_LENGTH
                        ]
                    ),
                    url=document.url,
                )
            )

        return compressed_documents