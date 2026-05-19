from memory.session_memory import (
    SessionMemory,
)
from memory.summary_memory import (
    SummaryMemory,
)
from state.schema import ResearchState


class MemoryManager:
    """Central conversational memory manager."""

    @staticmethod
    def update_memory(
        state: ResearchState,
        user_query: str,
        assistant_response: str,
    ) -> None:
        SessionMemory.add_user_message(
            state,
            user_query,
        )

        SessionMemory.add_ai_message(
            state,
            assistant_response,
        )

        SessionMemory.increment_turn(
            state,
        )

        should_summarize = (
            SummaryMemory.should_summarize(
                state,
            )
        )

        if should_summarize:
            state["conversation_summary"] = (
                SummaryMemory
                .update_summary(
                    state,
                )
            )