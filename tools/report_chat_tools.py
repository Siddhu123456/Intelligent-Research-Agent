from itertools import chain

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

from tools.prompts.report_chat_tools_prompts import (
    REPORT_CHAT_SYSTEM_PROMPT,
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

        #  SUMMARY MODE

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
                        REPORT_CHAT_SYSTEM_PROMPT,
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

        raw_response = (
            response.content
        )

        # Defensive normalization

        if not isinstance(
            raw_response,
            str,
        ):

            raw_response = str(
                raw_response
            )

        raw_response = (
            raw_response.strip()
        )

        # Try structured extraction first

        parsed_response = (
            JSONParser.safe_extract(
                content=raw_response,
                fallback=None,
            )
        )

        # Use parsed JSON answer
        # if available

        if (
            isinstance(
                parsed_response,
                dict,
            )
            and parsed_response.get(
                "answer"
            )
        ):

            answer = str(
                parsed_response.get(
                    "answer",
                    "",
                )
            )

        # Otherwise fallback to
        # cleaned raw response

        else:

            answer = (
                JSONParser
                .clean_response(
                    raw_response
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
        
    @staticmethod
    async def stream_report_answer(
        state,
        question: str,
        conversation_context: str,
    ):
        """
        Stream conversational report answers
        token-by-token using semantic RAG
        and intelligent workspace routing.
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

            yield (
                "No relevant report context "
                "was found for this question."
            )

            return

        llm = (
            LLMFactory
            .create_qwen_llm(
                temperature=0.2,
                streaming=True,
            )
        )

        prompt = (
            ChatPromptTemplate
            .from_messages(
                [
                    (
                        "system",
                        REPORT_CHAT_SYSTEM_PROMPT,
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

        async for chunk in chain.astream(
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
        ):

            if hasattr(
                chunk,
                "content",
            ):

                token = chunk.content

                if token:

                    yield token