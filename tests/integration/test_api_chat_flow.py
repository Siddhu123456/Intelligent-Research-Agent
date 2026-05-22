import importlib
import sys
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from api.session_manager import SessionManager
from fastapi.testclient import TestClient


def _build_test_client_with_executor(run_with_state_mock):
    """Reload the FastAPI app while injecting a fake GraphExecutor."""

    SessionManager._sessions.clear()

    fake_executor = MagicMock()
    fake_executor.run_with_state = run_with_state_mock

    sys.modules.pop("api.routes", None)
    sys.modules.pop("api.app", None)

    with patch("graph.executor.GraphExecutor", return_value=fake_executor):
        module = importlib.import_module("api.app")

    return TestClient(module.app)


class TestApiChatFlow:
    def test_chat_endpoint_creates_new_session(self):
        """Test chat endpoint returns a final report and stores session state."""

        fake_state = {
            "session_id": "session_123",
            "final_report": "This is a mocked report.",
        }

        client = _build_test_client_with_executor(
            AsyncMock(return_value=fake_state),
        )

        response = client.post(
            "/chat",
            json={
                "query": "What is the future of AI research?",
            },
        )

        assert response.status_code == 200
        payload = response.json()

        assert payload["response"] == "This is a mocked report."
        assert payload["session_id"]
        assert (
            SessionManager.get_session(payload["session_id"])["final_report"]
            == "This is a mocked report."
        )

    def test_chat_endpoint_reuses_existing_session(self):
        """Test chat endpoint updates an existing session state."""

        session_id = "existing-session"
        SessionManager._sessions.clear()
        SessionManager.save_session(
            session_id=session_id,
            state={
                "session_id": session_id,
                "query": "old query",
                "final_report": "old report",
            },
        )

        fake_state = {
            "session_id": session_id,
            "query": "new query",
            "final_report": "updated report",
        }

        client = _build_test_client_with_executor(
            AsyncMock(return_value=fake_state),
        )

        response = client.post(
            "/chat",
            json={
                "session_id": session_id,
                "query": "new query",
            },
        )

        assert response.status_code == 200
        payload = response.json()

        assert payload["session_id"] == session_id
        assert payload["response"] == "updated report"
        assert (
            SessionManager.get_session(session_id)["query"]
            == "new query"
        )
