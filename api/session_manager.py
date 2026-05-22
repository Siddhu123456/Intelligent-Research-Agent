from collections import defaultdict

from state.schema import ResearchState

from utils.vector_store import VectorStoreManager


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

    @classmethod
    def end_session(
        cls,
        session_id: str,
        clear_chroma: bool = True,
    ) -> None:
        """Remove an in-memory session and optionally clear Chroma files.

        Use this to perform per-session cleanup. `clear_chroma=True` will
        attempt to delete the `./chroma_db` directory via
        `VectorStoreManager.clear_persisted_data()`.
        """

        if session_id in cls._sessions:
            try:
                del cls._sessions[session_id]
            except Exception:
                pass

        if clear_chroma:
            try:
                VectorStoreManager.clear_persisted_data()
            except Exception:
                pass