from typing import (
    Literal,
    TypedDict,
)

from langgraph.graph.message import (
    add_messages,
)

from typing_extensions import (
    Annotated,
)

from state.models import (
    Citation,
    Document,
    SubQuery,
)


class ResearchState(
    TypedDict,
):
    """Shared state across all graph nodes."""

    # Core query state

    query: str

    contextualized_query: str

    # Workflow execution state

    current_step: str

    workflow_decision: str

    retry_count: int

    max_retries: int

    low_confidence: bool

    error: str | None

    # Retrieval state

    sub_queries: list[
        SubQuery
    ]

    retrieved_documents: list[
        Document
    ]

    # Analysis state

    analysis_summary: str

    key_findings: list[str]

    citations: list[
        Citation
    ]

    # Report workspace state

    active_report: str

    refinement_query: str

    report_chat_query: str

    report_chat_response: str

    report_history: list[
        dict
    ]

    report_version_history: list[
        dict
    ]

    # Document generation state

    generated_pdf: bytes | None

    # Conversation memory state

    messages: Annotated[
        list,
        add_messages,
    ]

    conversation_summary: str

    conversation_turn: int

    # Session state

    session_id: str

    # Workspace mode

    mode: Literal[
        "REPORT_GENERATION",
        "REPORT_CHAT",
        "REPORT_REFINEMENT",
        "DOCUMENT_GENERATION",
    ]

    # Execution metadata

    run_metadata: dict[
        str,
        str,
    ]