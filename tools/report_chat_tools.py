from langchain_core.prompts import (
    ChatPromptTemplate,
)

from utils.json_parser import (
    JSONParser,
)

from utils.llm_factory import (
    LLMFactory,
)


class ReportChatTools:
    """Tools for conversational report Q&A."""

    MAX_RESPONSE_LENGTH = 4000

    MAX_CONTEXT_LENGTH = 4000

    @staticmethod
    def answer_report_question(
        report: str,
        question: str,
    ) -> str:
        """Answer questions using compressed workspace memory."""

        if not report.strip():

            return (
                "No active report is available "
                "for question answering."
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
                            "research workspace "
                            "assistant.\n\n"

                            "You are provided with "
                            "compressed conversational "
                            "workspace memory derived "
                            "from a larger research report.\n\n"

                            "Your task is to answer "
                            "questions using ONLY the "
                            "provided workspace memory.\n\n"

                            "IMPORTANT RULES:\n"
                            "- answer ONLY from the "
                            "workspace memory\n"
                            "- do not hallucinate\n"
                            "- do not invent facts\n"
                            "- if information is missing, "
                            "clearly say so\n"
                            "- preserve technical accuracy\n"
                            "- provide concise and "
                            "professional answers\n"
                            "- infer relationships only "
                            "when strongly supported\n"
                            "- optimize for conversational "
                            "workspace continuity\n\n"

                            "The workspace memory may be:\n"
                            "- compressed\n"
                            "- summarized\n"
                            "- condensed\n\n"

                            "So prioritize:\n"
                            "- key findings\n"
                            "- important concepts\n"
                            "- technical insights\n"
                            "- conclusions\n"
                            "- important terminology\n\n"

                            "Return ONLY valid JSON.\n\n"

                            "Format:\n"
                            "{{\n"
                            '  "answer": "..."\n'
                            "}}"
                        ),
                    ),
                    (
                        "human",
                        (
                            "Compressed Workspace "
                            "Memory:\n"
                            "{report}\n\n"

                            "Question:\n"
                            "{question}"
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
                    :ReportChatTools
                    .MAX_CONTEXT_LENGTH
                ],
                "question": question,
            }
        )

        parsed_response = (
            JSONParser.safe_extract(
                content=(
                    response.content
                ),
                fallback={
                    "answer": (
                        "Unable to generate "
                        "a grounded answer "
                        "from the workspace "
                        "memory."
                    ),
                },
            )
        )

        answer = (
            parsed_response.get(
                "answer",
                "",
            )
        )

        # Defensive validation

        if not isinstance(
            answer,
            str,
        ):

            answer = str(
                answer
            )

        return (
            answer.strip()[
                :ReportChatTools
                .MAX_RESPONSE_LENGTH
            ]
        )