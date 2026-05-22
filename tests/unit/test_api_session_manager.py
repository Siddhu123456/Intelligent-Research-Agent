from api.session_manager import SessionManager


def test_session_manager_save_and_get():
    SessionManager._sessions.clear()

    state = {
        "session_id": "session_1",
        "query": "test",
    }

    SessionManager.save_session(
        session_id="session_1",
        state=state,
    )

    assert (
        SessionManager.get_session("session_1")
        == state
    )


def test_end_session_removes_state(monkeypatch):
    SessionManager._sessions.clear()

    SessionManager.save_session(
        session_id="session_2",
        state={"session_id": "session_2"},
    )

    monkeypatch.setattr(
        "api.session_manager.VectorStoreManager.clear_persisted_data",
        lambda: None,
    )

    SessionManager.end_session(
        session_id="session_2",
        clear_chroma=True,
    )

    assert (
        SessionManager.get_session("session_2")
        is None
    )
