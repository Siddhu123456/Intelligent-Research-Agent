from utils.langsmith_wrapper import traceable

from memory.session_memory import (
    SessionMemory,
)

from state.constants import (
    CurrentStep,
)

from state.schema import (
    ResearchState,
)

from tools.context_tools import (
    ContextTools,
)

from utils.logger import (
    setup_logger,
)


logger = setup_logger(
    __name__,
)


class ContextAgent:
    """Agent responsible for query contextualization."""

    MIN_CONTEXT_MESSAGES = 2

    @staticmethod
    @traceable(
        name="context_agent",
    )
    def run(
        state: ResearchState,
    ) -> ResearchState:

        try:

            logger.info(
                "Starting contextual query rewrite",
            )

            query = state["query"]

            recent_messages = (
                SessionMemory
                .get_recent_messages(
                    state,
                )
            )

            # Skip rewriting for fresh sessions

            if (
                len(recent_messages)
                < ContextAgent
                .MIN_CONTEXT_MESSAGES
            ):

                state[
                    "contextualized_query"
                ] = query

                state["current_step"] = (
                    CurrentStep
                    .CONTEXTUALIZED
                    .value
                )

                logger.info(
                    "Skipping query rewrite "
                    "for fresh session",
                )

                return state

            conversation_context = (
                ContextTools
                .build_conversation_context(
                    messages=(
                        recent_messages
                    ),
                    summary=(
                        state[
                            "conversation_summary"
                        ]
                    ),
                )
            )

            rewritten_query = (
                ContextTools
                .rewrite_query(
                    query=query,
                    conversation_context=(
                        conversation_context
                    ),
                )
            )

            state[
                "contextualized_query"
            ] = rewritten_query

            state["current_step"] = (
                CurrentStep
                .CONTEXTUALIZED
                .value
            )

            logger.info(
                "Contextualized query: %s",
                rewritten_query,
            )

            return state

        except Exception as error:

            logger.error(
                "Context agent failed: %s",
                str(error),
            )

            state[
                "contextualized_query"
            ] = (
                state["query"]
            )

            state["error"] = str(
                error,
            )

            return state