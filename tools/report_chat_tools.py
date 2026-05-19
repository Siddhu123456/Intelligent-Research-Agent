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

    @staticmethod
    def answer_report_question(
        report: str,
        question: str,
    ) -> str:
        """Answer questions using report context."""

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
                            "research assistant.\n\n"

                            "Your task is to answer "
                            "questions strictly using "
                            "the provided research "
                            "report.\n\n"

                            "IMPORTANT RULES:\n"
                            "- answer ONLY from the "
                            "report context\n"
                            "- do not hallucinate\n"
                            "- do not invent facts\n"
                            "- if information is missing, "
                            "clearly say so\n"
                            "- provide concise and "
                            "professional answers\n"
                            "- preserve technical accuracy\n"
                            "- summarize relevant sections "
                            "when necessary\n\n"

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
                            "Research Report:\n"
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
                "report": report,
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
                        "from the report."
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

        return (
            answer.strip()[
                :ReportChatTools
                .MAX_RESPONSE_LENGTH
            ]
        )