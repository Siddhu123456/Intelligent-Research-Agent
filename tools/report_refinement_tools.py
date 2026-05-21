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

    MAX_SECTION_LENGTH = 4000

    MAX_REFINEMENT_QUERY = 1000

    MAX_SECTION_CONTEXT = 2500

    @staticmethod
    def classify_refinement_intent(
        refinement_query: str,
        existing_sections: list[str],
    ) -> dict:
        """
        Detect refinement intent safely.
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

                            "Determine whether the "
                            "user wants to:\n"
                            "- modify existing section\n"
                            "- add new section\n\n"

                            "IMPORTANT RULES:\n"

                            "- if user asks to ADD, "
                            "CREATE, INSERT, INCLUDE "
                            "or introduce a NEW topic "
                            "not already present, "
                            "prefer add_section\n\n"

                            "- for modify_section, "
                            "target_section MUST "
                            "exactly match one of "
                            "the available sections\n\n"

                            "- NEVER invent section names\n"

                            "- use snake_case only\n"

                            "- return ONLY valid JSON\n\n"

                            "Available Sections:\n"
                            "{sections}\n\n"

                            "Format:\n"
                            "{{\n"
                            '  "intent": "modify_section",\n'
                            '  "target_section": '
                            '"analysis_and_insights"\n'
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
                "query":
                refinement_query[
                    :ReportRefinementTools
                    .MAX_REFINEMENT_QUERY
                ],

                "sections":
                ", ".join(
                    existing_sections
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
                    "analysis_and_insights",
                },
            )
        )

        target_section = (
            parsed_response.get(
                "target_section",
                "",
            )
        )

        intent = (
            parsed_response.get(
                "intent",
                "modify_section",
            )
        )

        # Safety correction

        if (
            intent == "modify_section"
            and target_section
            not in existing_sections
        ):

            parsed_response[
                "target_section"
            ] = (
                "analysis_and_insights"
            )

        return parsed_response

    @staticmethod
    def refine_section(
        section_name: str,
        section_content: str,
        refinement_query: str,
    ) -> str:
        """
        Refine ONLY one section.
        """

        section_content = (
            section_content[
                :ReportRefinementTools
                .MAX_SECTION_CONTEXT
            ]
        )

        refinement_query = (
            refinement_query[
                :ReportRefinementTools
                .MAX_REFINEMENT_QUERY
            ]
        )

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

                            "Refine ONLY the provided "
                            "report section.\n\n"

                            "IMPORTANT RULES:\n"
                            "- preserve meaning\n"
                            "- preserve accuracy\n"
                            "- improve clarity\n"
                            "- avoid hallucinations\n"
                            "- return ONLY updated section\n"
                            "- do NOT generate full report\n\n"

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
        existing_sections: list[str],
    ) -> str:
        """
        Generate lightweight new section.
        """

        refinement_query = (
            refinement_query[
                :ReportRefinementTools
                .MAX_REFINEMENT_QUERY
            ]
        )

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

                            "Generate ONLY the requested "
                            "new report section.\n\n"

                            "IMPORTANT RULES:\n"
                            "- generate section content only\n"
                            "- no markdown titles\n"
                            "- concise but informative\n"
                            "- professional tone\n"
                            "- avoid hallucinations\n"
                            "- return ONLY valid JSON\n\n"

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

                            "Existing Sections:\n"
                            "{existing_sections}\n\n"

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

                "existing_sections":
                ", ".join(
                    existing_sections
                ),

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

    @staticmethod
    def determine_section_placement(
        new_section: str,
        existing_sections: list[str],
    ) -> dict:
        """
        Determine intelligent placement.
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
                            "Determine where a NEW "
                            "section should be placed "
                            "inside a research report.\n\n"

                            "Return ONLY valid JSON.\n\n"

                            "Format:\n"
                            "{{\n"
                            '  "insert_before": "conclusion"\n'
                            "}}"
                        ),
                    ),
                    (
                        "human",
                        (
                            "Existing Sections:\n"
                            "{existing_sections}\n\n"

                            "New Section:\n"
                            "{new_section}"
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
                "new_section":
                new_section,

                "existing_sections":
                ", ".join(
                    existing_sections
                ),
            }
        )

        parsed_response = (
            JSONParser.safe_extract(
                content=(
                    response.content
                ),
                fallback={
                    "insert_before":
                    "conclusion",
                },
            )
        )

        return parsed_response