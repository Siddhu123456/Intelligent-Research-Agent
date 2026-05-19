from memory.session_memory import (
    SessionMemory,
)

from memory.summary_memory import (
    SummaryMemory,
)

from state.schema import (
    ResearchState,
)


class MemoryManager:
    """Central conversational memory manager."""

    @staticmethod
    def update_memory(
        state: ResearchState,
        user_query: str,
        assistant_response: str,
    ) -> None:
        """Update conversational memory state."""

        # Store user message

        SessionMemory.add_user_message(
            state,
            user_query,
        )

        # Store assistant response

        SessionMemory.add_ai_message(
            state,
            assistant_response,
        )

        # Increment conversation turn

        SessionMemory.increment_turn(
            state,
        )

        # Update long-term summary memory

        if (
            SummaryMemory.should_summarize(
                state,
            )
        ):

            updated_summary = (
                SummaryMemory
                .update_summary(
                    state,
                )
            )

            if updated_summary:

                state[
                    "conversation_summary"
                ] = (
                    updated_summary
                )