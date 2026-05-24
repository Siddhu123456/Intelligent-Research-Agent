from langchain_core.prompts import (
    ChatPromptTemplate,
)

from tools.report_vector_tools import (
    ReportVectorTools,
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
        state,
        question: str,
        conversation_context: str,
    ) -> str:
        """
        Answer questions using
        conversational semantic RAG
        with intelligent workspace
        context routing.
        """

        session_id = (
            state.get(
                "session_id",
                "",
            )
        )

        report_sections = (
            state.get(
                "report_sections",
                {},
            )
        )
        
        print(state["report_sections"])

        compressed_report_context = (
            state.get(
                "compressed_report_context",
                "",
            )
        )

        # Defensive normalization

        if not isinstance(
            conversation_context,
            str,
        ):

            conversation_context = str(
                conversation_context
            )

        if not isinstance(
            compressed_report_context,
            str,
        ):

            compressed_report_context = str(
                compressed_report_context
            )

        lowered_question = (
            question.lower()
        )

        section_keywords = [
            "section",
            "sections",
            "structure",
            "outline",
        ]

        summary_keywords = [
            "summary",
            "summarize",
            "overview",
            "main findings",
        ]

        # SECTION MODE

        if any(
            keyword in lowered_question
            for keyword in section_keywords
        ):

            ordered_sections = list(
                report_sections.keys()
            )

            report_context = (
                f"The report contains "
                f"{len(ordered_sections)} "
                f"sections.\n\n"

                "Available Report Sections:\n\n"
                + "\n".join(
                    [
                        (
                            f"- {section}"
                        )
                        for section in (
                            ordered_sections
                        )
                    ]
                )
            )

        # SUMMARY MODE

        elif any(
            keyword in lowered_question
            for keyword in summary_keywords
        ):

            report_context = (
                compressed_report_context
            )

        # DEFAULT → SEMANTIC RAG

        else:

            retrieved_sections = (
                ReportVectorTools
                .semantic_report_search(
                    query=question,
                    session_id=session_id,
                )
            )

            report_context = "\n\n".join(
                [
                    (
                        f"Section: "
                        f"{section['section']}\n\n"
                        f"{section['content']}"
                    )
                    for section in (
                        retrieved_sections
                    )
                ]
            )

        # Defensive normalization

        if not isinstance(
            report_context,
            str,
        ):

            report_context = str(
                report_context
            )

        # Validate retrieval results

        if not report_context.strip():

            return (
                "No relevant report context "
                "was found for this question."
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

                            "You are provided with:\n"

                            "1. conversational workspace "
                            "context\n"

                            "2. report workspace context\n\n"

                            "Use BOTH to answer "
                            "user questions accurately.\n\n"

                            "IMPORTANT RULES:\n"

                            "- answer ONLY from the "
                            "provided report context\n"

                            "- do not hallucinate\n"

                            "- do not invent facts\n"

                            "- if information is missing, "
                            "clearly say so\n"

                            "- preserve technical accuracy\n"

                            "- provide concise and "
                            "professional answers\n"

                            "- infer relationships only "
                            "when strongly supported\n"

                            "- mention relevant section "
                            "names when appropriate\n"

                            "- optimize for grounded "
                            "conversational Q&A\n\n"

                            "- use conversation context "
                            "to resolve references like:\n"
                            "  it, they, this, that\n"

                            "- maintain conversational "
                            "continuity\n\n"

                            "- if the question asks about "
                            "report structure or sections, "
                            "use available section context\n"

                            "- if the question asks for "
                            "summary or overview, use "
                            "compressed report summary\n"
                            
                            "- if section information is "
                            "explicitly available in the "
                            "context, directly use it\n"

                            "- otherwise use semantically "
                            "retrieved report context\n\n"

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
                            "Conversation Context:\n"
                            "{conversation_context}\n\n"

                            "Report Workspace Context:\n"
                            "{report_context}\n\n"

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
                "conversation_context":
                conversation_context[
                    :ReportChatTools
                    .MAX_CONTEXT_LENGTH
                ],

                "report_context":
                report_context[
                    :ReportChatTools
                    .MAX_CONTEXT_LENGTH
                ],

                "question":
                question,
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
                        "from the report "
                        "workspace context."
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

        # Defensive normalization

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