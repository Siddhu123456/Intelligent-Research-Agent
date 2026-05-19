from typing import TypedDict

from langgraph.graph.message import add_messages
from typing_extensions import Annotated

from state.models import Citation
from state.models import Document
from state.models import SubQuery


class ResearchState(TypedDict):
    """Shared state across all graph nodes."""

    query: str

    sub_queries: list[SubQuery]

    retrieved_documents: list[Document]

    analysis_summary: str

    key_findings: list[str]

    citations: list[Citation]

    final_report: str

    current_step: str

    low_confidence: bool

    error: str | None

    messages: Annotated[list, add_messages]
    
    conversation_summary: str

    contextualized_query: str

    session_id: str

    conversation_turn: int
    
    retry_count: int

    max_retries: int

    workflow_decision: str

    run_metadata: dict[str, str]