from langchain_core.prompts import (
    ChatPromptTemplate,
)

from utils.json_parser import (
    JSONParser,
)

from utils.llm_factory import (
    LLMFactory,
)


class ReportCompressionTools:
    """Tools for conversational workspace compression."""

    MAX_COMPRESSED_LENGTH = 4000

    @staticmethod
    def compress_report(
        report: str,
    ) -> str:
        """
        Compress a full report into
        conversational workspace memory.

        This compressed memory is optimized for:
        - report chat
        - workspace continuity
        - low token usage
        - fast reasoning
        """

        if not report.strip():

            return (
                "No report available "
                "for compression."
            )

        llm = (
            LLMFactory
            .create_qwen_llm(
                temperature=0.1,
            )
        )

        prompt = (
            ChatPromptTemplate
            .from_messages(
                [
                    (
                        "system",
                        (
                            "You are an expert "
                            "workspace memory "
                            "compression agent.\n\n"

                            "Your task is to compress "
                            "a research report into "
                            "high-quality conversational "
                            "workspace memory.\n\n"

                            "IMPORTANT RULES:\n"
                            "- preserve important findings\n"
                            "- preserve technical insights\n"
                            "- preserve conclusions\n"
                            "- preserve important terminology\n"
                            "- preserve overall report meaning\n"
                            "- remove repetition\n"
                            "- remove verbose explanations\n"
                            "- remove unnecessary formatting\n"
                            "- optimize for future Q&A\n"
                            "- optimize for low token usage\n"
                            "- preserve factual accuracy\n"
                            "- avoid hallucinations\n\n"

                            "The output should behave like:\n"
                            "- compressed workspace memory\n"
                            "- NOT a simple summary\n\n"

                            "The compressed memory should contain:\n"
                            "- major concepts\n"
                            "- important findings\n"
                            "- technical observations\n"
                            "- key conclusions\n"
                            "- critical context\n"
                            "- domain terminology\n\n"

                            "Return ONLY valid JSON.\n\n"

                            "Format:\n"
                            "{{\n"
                            '  "compressed_context": "..."\n'
                            "}}"
                        ),
                    ),
                    (
                        "human",
                        (
                            "Research Report:\n"
                            "{report}"
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
                "report": report[
                    :4000
                ],
            }
        )

        parsed_response = (
            JSONParser.safe_extract(
                content=(
                    response.content
                ),
                fallback={
                    "compressed_context":
                    report[:4000]
                },
            )
        )

        compressed_context = (
            parsed_response.get(
                "compressed_context",
                report[:4000],
            )
        )

        # Defensive validation

        if not isinstance(
            compressed_context,
            str,
        ):

            compressed_context = str(
                compressed_context
            )

        return (
            compressed_context
            .strip()[
                :ReportCompressionTools
                .MAX_COMPRESSED_LENGTH
            ]
        )
        
    @staticmethod
    def update_compressed_context(
        previous_context: str,
        updated_section_name: str,
        updated_section_content: str,
        refinement_query: str,
    ) -> str:
        """
        Incrementally update compressed
        conversational workspace memory.

        This avoids expensive full-report
        recompression after every refinement.
        """

        llm = (
            LLMFactory
            .create_qwen_llm(
                temperature=0.1,
            )
        )

        previous_context = (
            previous_context[:2500]
        )

        updated_section_content = (
            updated_section_content[:1500]
        )

        refinement_query = (
            refinement_query[:800]
        )

        prompt = (
            ChatPromptTemplate
            .from_messages(
                [
                    (
                        "system",
                        (
                            "You are an expert "
                            "workspace memory "
                            "compression agent.\n\n"

                            "Your task is to update "
                            "compressed conversational "
                            "workspace memory after "
                            "a report refinement "
                            "operation.\n\n"

                            "IMPORTANT RULES:\n"
                            "- preserve existing memory\n"
                            "- integrate ONLY relevant updates\n"
                            "- preserve important findings\n"
                            "- preserve technical concepts\n"
                            "- preserve conclusions\n"
                            "- preserve terminology\n"
                            "- avoid repetition\n"
                            "- avoid hallucinations\n"
                            "- optimize for low token usage\n"
                            "- maintain conversational continuity\n"
                            "- generate concise memory\n\n"

                            "The output should behave like:\n"
                            "- persistent workspace memory\n"
                            "- conversational research memory\n"
                            "- compressed report context\n\n"

                            "Return ONLY valid JSON.\n\n"

                            "Format:\n"
                            "{{\n"
                            '  "updated_context": "..."\n'
                            "}}"
                        ),
                    ),
                    (
                        "human",
                        (
                            "Previous Workspace "
                            "Memory:\n"
                            "{previous_context}\n\n"

                            "Updated Section Name:\n"
                            "{section_name}\n\n"

                            "Updated Section Content:\n"
                            "{section_content}\n\n"

                            "Refinement Operation:\n"
                            "{refinement_query}"
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
                "previous_context":
                previous_context,

                "section_name":
                updated_section_name,

                "section_content":
                updated_section_content,

                "refinement_query":
                refinement_query,
            }
        )

        parsed_response = (
            JSONParser.safe_extract(
                content=(
                    response.content
                ),
                fallback={
                    "updated_context":
                    (
                        previous_context
                        + "\n\n"
                        + updated_section_name
                        + ":\n"
                        + updated_section_content[
                            :1000
                        ]
                    ),
                },
            )
        )

        updated_context = (
            parsed_response.get(
                "updated_context",
                previous_context,
            )
        )

        if not isinstance(
            updated_context,
            str,
        ):

            updated_context = str(
                updated_context
            )

        return (
            updated_context
            .strip()[:4000]
        )