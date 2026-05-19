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
    """Tools for intelligent report refinement."""

    MAX_REPORT_LENGTH = 12000

    @staticmethod
    def refine_report(
        report: str,
        refinement_instruction: str,
    ) -> str:
        """Refine existing report intelligently."""

        if not report.strip():

            return (
                "No active report available "
                "for refinement."
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

                            "Your task is to refine "
                            "and improve an existing "
                            "research report using "
                            "the provided refinement "
                            "instruction.\n\n"

                            "IMPORTANT RULES:\n"
                            "- preserve report structure\n"
                            "- preserve factual accuracy\n"
                            "- preserve technical quality\n"
                            "- improve clarity and depth\n"
                            "- integrate refinement naturally\n"
                            "- avoid hallucinations\n"
                            "- avoid removing important content\n"
                            "- maintain professional tone\n"
                            "- return the FULL updated report\n\n"

                            "Return ONLY valid JSON.\n\n"

                            "Format:\n"
                            "{{\n"
                            '  "refined_report": "..."\n'
                            "}}"
                        ),
                    ),
                    (
                        "human",
                        (
                            "Existing Report:\n"
                            "{report}\n\n"

                            "Refinement Instruction:\n"
                            "{instruction}"
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
                "report": report,
                "instruction": (
                    refinement_instruction
                ),
            }
        )

        parsed_response = (
            JSONParser.safe_extract(
                content=(
                    response.content
                ),
                fallback={
                    "refined_report": report,
                },
            )
        )

        refined_report = (
            parsed_response.get(
                "refined_report",
                report,
            )
        )

        return (
            refined_report.strip()[
                :ReportRefinementTools
                .MAX_REPORT_LENGTH
            ]
        )