import re

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

            prompt = (
                ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            (
                                "You are a contextual "
                                "compression agent.\n\n"
                                "Extract ONLY information "
                                "relevant to the user query.\n\n"
                                "Rules:\n"
                                "- remove irrelevant details\n"
                                "- preserve factual accuracy\n"
                                "- preserve technical details\n"
                                "- avoid hallucinations\n"
                                "- avoid summaries unrelated "
                                "to the query\n"
                                "- keep the response concise\n"
                                "- return factual compressed "
                                "content only\n\n"
                                "Do NOT include reasoning.\n"
                                "Do NOT include "
                                "<think> tags.\n\n"
                                "Return ONLY valid JSON.\n\n"
                                "Format:\n"
                                "{{\n"
                                '  "content": "..." \n'
                                "}}"
                            ),
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
                            :4000
                        ]
                    ),
                }
            )

            try:

                parsed_response = (
                    JSONParser.extract_json(
                        response.content,
                    )
                )

                compressed_content = (
                    parsed_response[
                        "content"
                    ]
                )

            except Exception:

                compressed_content = (
                    response.content
                )

            compressed_documents.append(
                Document(
                    source=document.source,
                    title=document.title,
                    content=(
                        compressed_content[
                            :4000
                        ].strip()
                    ),
                    url=document.url,
                )
            )

        return compressed_documents