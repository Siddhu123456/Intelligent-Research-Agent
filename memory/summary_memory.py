from langchain_core.prompts import (
    ChatPromptTemplate,
)

from memory.session_memory import (
    SessionMemory,
)

from state.schema import (
    ResearchState,
)

from utils.json_parser import (
    JSONParser,
)

from utils.llm_factory import (
    LLMFactory,
)

from memory.prompts.summary_memory_prompts import (
    SUMMARY_MEMORY_SYSTEM_PROMPT,
)


class SummaryMemory:
    """Maintains compressed conversation memory."""

    SUMMARY_TRIGGER_TURN = 5

    MAX_MESSAGE_HISTORY = 10

    @staticmethod
    def should_summarize(
        state: ResearchState,
    ) -> bool:
        return (
            state["conversation_turn"]
            >= SummaryMemory
            .SUMMARY_TRIGGER_TURN
        )

    @staticmethod
    def update_summary(
        state: ResearchState,
    ) -> str:
        llm = (
            LLMFactory.create_qwen_llm(
                temperature=0.1,
            )
        )

        messages = (
            SessionMemory
            .get_recent_messages(
                state,
            )[
                -SummaryMemory
                .MAX_MESSAGE_HISTORY:
            ]
        )

        conversation_text = "\n".join(
            [
                (
                    f"{message.type.upper()}: "
                    f"{message.content}"
                )
                for message in messages
            ]
        )

        existing_summary = (
            state.get(
                "conversation_summary",
                "",
            )
        )

        prompt = (
            ChatPromptTemplate.from_messages(
                [
                    ("system",
                        SUMMARY_MEMORY_SYSTEM_PROMPT,
                    ),
                    (
                        "human",
                        (
                            "Existing Summary:\n"
                            "{existing_summary}\n\n"
                            "Recent Conversation:\n"
                            "{conversation_text}"
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
                "existing_summary": (
                    existing_summary
                ),
                "conversation_text": (
                    conversation_text
                ),
            }
        )

        try:

            parsed_response = (
                JSONParser.extract_json(
                    response.content,
                )
            )

            return (
                parsed_response[
                    "summary"
                ].strip()
            )

        except Exception:

            return (
                existing_summary
            )