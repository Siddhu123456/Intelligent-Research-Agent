from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from state.schema import (
    ResearchState,
)


class SessionMemory:
    """Manages conversational session memory."""

    MAX_RECENT_MESSAGES = 10

    @staticmethod
    def add_user_message(
        state: ResearchState,
        message: str,
    ) -> None:
        """Store user message."""

        if not message.strip():
            return

        state["messages"].append(
            HumanMessage(
                content=message.strip(),
            )
        )

    @staticmethod
    def add_ai_message(
        state: ResearchState,
        message: str,
    ) -> None:
        """Store assistant message."""
        # Defensive: coerce non-string inputs to string to avoid
        # AttributeError when message is a dict or other type.
        if not isinstance(message, str):
            try:
                message = str(message)
            except Exception:
                return

        if not message.strip():
            return

        state["messages"].append(
            AIMessage(
                content=message.strip(),
            )
        )

    @staticmethod
    def get_recent_messages(
        state: ResearchState,
    ) -> list:
        """Return recent conversational context."""

        return state[
            "messages"
        ][
            -SessionMemory
            .MAX_RECENT_MESSAGES:
        ]

    @staticmethod
    def increment_turn(
        state: ResearchState,
    ) -> None:
        """Increment conversation turn."""

        state[
            "conversation_turn"
        ] += 1

    @staticmethod
    def clear_messages(
        state: ResearchState,
    ) -> None:
        """Clear session messages."""

        state["messages"] = []