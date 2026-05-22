import pytest

from api.models import ChatRequest
from api.models import ChatResponse


def test_chat_request_requires_query():
    with pytest.raises(Exception):
        ChatRequest()


def test_chat_response_fields():
    response = ChatResponse(
        session_id="session_abc",
        response="Mock response",
    )

    assert response.session_id == "session_abc"
    assert response.response == "Mock response"
