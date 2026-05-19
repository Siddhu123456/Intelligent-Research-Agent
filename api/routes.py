from uuid import uuid4

from fastapi import APIRouter

from api.models import ChatRequest
from api.models import ChatResponse
from api.session_manager import (
    SessionManager,
)
from graph.executor import GraphExecutor
from utils.state_factory import (
    StateFactory,
)


router = APIRouter()

executor = GraphExecutor()


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
) -> ChatResponse:
    session_id = (
        request.session_id
        or str(uuid4())
    )

    existing_state = (
        SessionManager.get_session(
            session_id,
        )
    )

    if existing_state:
        state = existing_state

        state["query"] = (
            request.query
        )

    else:
        state = (
            StateFactory
            .create_initial_state(
                query=request.query,
            )
        )

        state["session_id"] = (
            session_id
        )

    result = await executor.run_with_state(
        state,
    )

    SessionManager.save_session(
        session_id=session_id,
        state=result,
    )

    return ChatResponse(
        session_id=session_id,
        response=result[
            "final_report"
        ],
    )