from langchain_core.messages import (
    AIMessage,
)
from langchain_core.messages import (
    HumanMessage,
)

from state.schema import ResearchState


class SessionMemory:
    """Manages conversational session memory."""

    MAX_RECENT_MESSAGES = 6

    @staticmethod
    def add_user_message(
        state: ResearchState,
        message: str,
    ) -> None:
        state["messages"].append(
            HumanMessage(
                content=message,
            )
        )

    @staticmethod
    def add_ai_message(
        state: ResearchState,
        message: str,
    ) -> None:
        state["messages"].append(
            AIMessage(
                content=message,
            )
        )

    @staticmethod
    def get_recent_messages(
        state: ResearchState,
    ) -> list:
        return state["messages"][
            -SessionMemory
            .MAX_RECENT_MESSAGES:
        ]

    @staticmethod
    def increment_turn(
        state: ResearchState,
    ) -> None:
        state["conversation_turn"] += 1