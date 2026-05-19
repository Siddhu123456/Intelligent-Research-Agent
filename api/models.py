from pydantic import BaseModel
from pydantic import Field


class ChatRequest(BaseModel):
    """Incoming chat request."""

    session_id: str | None = Field(
        default=None,
    )

    query: str


class ChatResponse(BaseModel):
    """Final chat response."""

    session_id: str

    response: str