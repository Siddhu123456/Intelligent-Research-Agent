from memory.session_memory import (
    SessionMemory,
)
from state.schema import ResearchState
from utils.llm_factory import (
    LLMFactory,
)


class SummaryMemory:
    """Maintains compressed conversation memory."""

    SUMMARY_TRIGGER_TURN = 5

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
            )
        )

        conversation_text = "\n".join(
            [
                message.content
                for message in messages
            ]
        )

        existing_summary = (
            state[
                "conversation_summary"
            ]
        )

        prompt = f"""
You are a conversation memory agent.

Existing Summary:
{existing_summary}

Recent Conversation:
{conversation_text}

Generate an updated concise summary
preserving:
- important research topics
- user intent
- previous conclusions
- ongoing context
"""

        response = llm.invoke(
            prompt,
        )

        return response.content