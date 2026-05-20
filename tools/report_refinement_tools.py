from langchain_core.prompts import (
    ChatPromptTemplate,
)

from utils.json_parser import (
    JSONParser,
)

from utils.llm_factory import (
    LLMFactory,
)


class ReportRefinementTools:
    """Tools for section-aware report refinement."""

    MAX_SECTION_LENGTH = 6000

    @staticmethod
    def classify_refinement_intent(
        refinement_query: str,
        existing_sections: list[str],
    ) -> dict:
        """
        Detect refinement intent.

        Determines:
        - add new section
        - modify existing section
        - target section
        """

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
                            "You are an intelligent "
                            "document refinement "
                            "classifier.\n\n"

                            "Your task is to classify "
                            "a report refinement request.\n\n"

                            "Determine:\n"
                            "- whether user wants to "
                            "MODIFY an existing section\n"
                            "- OR ADD a new section\n"
                            "- identify target section\n\n"

                            "IMPORTANT RULES:\n"
                            "- prefer MODIFY if user "
                            "references existing content\n"
                            "- prefer ADD if user asks "
                            "for a completely new topic\n"
                            "- target_section must be "
                            "normalized snake_case\n"
                            "- preserve semantic meaning\n"
                            "- return ONLY valid JSON\n\n"

                            "Available Sections:\n"
                            "{sections}\n\n"

                            "Output Format:\n"
                            "{{\n"
                            '  "intent": "modify_section",\n'
                            '  "target_section": "analysis"\n'
                            "}}\n\n"

                            "OR\n\n"

                            "{{\n"
                            '  "intent": "add_section",\n'
                            '  "target_section": "future_trends"\n'
                            "}}"
                        ),
                    ),
                    (
                        "human",
                        (
                            "Refinement Request:\n"
                            "{query}"
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
                "query": (
                    refinement_query
                ),
                "sections": (
                    ", ".join(
                        existing_sections
                    )
                ),
            }
        )

        parsed_response = (
            JSONParser.safe_extract(
                content=(
                    response.content
                ),
                fallback={
                    "intent":
                    "modify_section",
                    "target_section":
                    "analysis",
                },
            )
        )

        return parsed_response

    @staticmethod
    def refine_section(
        section_name: str,
        section_content: str,
        refinement_query: str,
    ) -> str:
        """
        Refine an existing section only.
        """

        llm = (
            LLMFactory
            .create_qwen_llm(
                temperature=0.2,
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
                            "research report editor.\n\n"

                            "Your task is to refine "
                            "ONLY the provided report "
                            "section.\n\n"

                            "IMPORTANT RULES:\n"
                            "- preserve existing meaning\n"
                            "- preserve factual accuracy\n"
                            "- improve clarity and depth\n"
                            "- integrate refinement naturally\n"
                            "- maintain professional tone\n"
                            "- avoid hallucinations\n"
                            "- return ONLY updated section\n"
                            "- do NOT generate entire report\n\n"

                            "Return ONLY valid JSON.\n\n"

                            "Format:\n"
                            "{{\n"
                            '  "updated_section": "..."\n'
                            "}}"
                        ),
                    ),
                    (
                        "human",
                        (
                            "Section Name:\n"
                            "{section_name}\n\n"

                            "Current Section:\n"
                            "{section_content}\n\n"

                            "Refinement Request:\n"
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
                "section_name":
                section_name,
                "section_content":
                section_content,
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
                    "updated_section":
                    section_content,
                },
            )
        )

        updated_section = (
            parsed_response.get(
                "updated_section",
                section_content,
            )
        )

        if not isinstance(
            updated_section,
            str,
        ):

            updated_section = str(
                updated_section
            )

        return (
            updated_section
            .strip()[
                :ReportRefinementTools
                .MAX_SECTION_LENGTH
            ]
        )

    @staticmethod
    def generate_new_section(
        section_name: str,
        report_topic: str,
        refinement_query: str,
    ) -> str:
        """
        Generate entirely new section.
        """

        llm = (
            LLMFactory
            .create_qwen_llm(
                temperature=0.2,
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
                            "research report writer.\n\n"

                            "Generate a NEW report "
                            "section.\n\n"

                            "IMPORTANT RULES:\n"
                            "- generate only the section\n"
                            "- preserve professional tone\n"
                            "- maintain factual accuracy\n"
                            "- integrate naturally with "
                            "research reports\n"
                            "- avoid hallucinations\n"
                            "- avoid markdown titles\n"
                            "- return ONLY section content\n\n"

                            "Return ONLY valid JSON.\n\n"

                            "Format:\n"
                            "{{\n"
                            '  "new_section": "..."\n'
                            "}}"
                        ),
                    ),
                    (
                        "human",
                        (
                            "Report Topic:\n"
                            "{topic}\n\n"

                            "New Section Name:\n"
                            "{section_name}\n\n"

                            "User Request:\n"
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
                "topic":
                report_topic,
                "section_name":
                section_name,
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
                    "new_section":
                    (
                        "This section discusses "
                        f"{section_name.replace('_', ' ')} "
                        "in the context of the "
                        "research topic."
                    ),
                },
            )
        )

        new_section = (
            parsed_response.get(
                "new_section",
                "",
            )
        )
        
        if not new_section:

            new_section = (
                "Additional discussion on "
                f"{section_name.replace('_', ' ')}."
            )

        if not isinstance(
            new_section,
            str,
        ):

            new_section = str(
                new_section
            )

        return (
            new_section
            .strip()[
                :ReportRefinementTools
                .MAX_SECTION_LENGTH
            ]
        )