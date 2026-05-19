from collections import defaultdict

from state.schema import ResearchState


class SessionManager:
    """In-memory session manager."""

    _sessions: dict[
        str,
        ResearchState,
    ] = defaultdict(dict)

    @classmethod
    def get_session(
        cls,
        session_id: str,
    ) -> ResearchState | None:
        return cls._sessions.get(
            session_id,
        )

    @classmethod
    def save_session(
        cls,
        session_id: str,
        state: ResearchState,
    ) -> None:
        cls._sessions[
            session_id
        ] = state