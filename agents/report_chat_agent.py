from langsmith import (
    traceable,
)

from memory.memory_manager import (
    MemoryManager,
)

from state.constants import (
    CurrentStep,
)

from state.schema import (
    ResearchState,
)

from tools.report_chat_tools import (
    ReportChatTools,
)

from utils.logger import (
    setup_logger,
)


logger = setup_logger(
    __name__,
)


class ReportChatAgent:
    """Agent responsible for report Q&A."""

    MAX_FALLBACK_REPORT_LENGTH = 4000

    @staticmethod
    @traceable(
        name="report_chat_agent",
    )
    def run(
        state: ResearchState,
    ) -> ResearchState:

        try:

            logger.info(
                "Starting report chat workflow",
            )

            compressed_report_context = (
                state.get(
                    "compressed_report_context",
                    "",
                )
            )

            active_report = (
                state.get(
                    "active_report",
                    "",
                )
            )

            report_chat_query = (
                state.get(
                    "report_chat_query",
                    "",
                )
            )

            # Prefer compressed workspace memory

            report_context = (
                compressed_report_context
            )

            # Fallback for older sessions

            if not report_context.strip():

                logger.warning(
                    "Compressed context missing. "
                    "Using fallback report slice.",
                )

                report_context = (
                    active_report[
                        :ReportChatAgent
                        .MAX_FALLBACK_REPORT_LENGTH
                    ]
                )

            # Validate report availability

            if not report_context.strip():

                logger.warning(
                    "No report context found",
                )

                state[
                    "report_chat_response"
                ] = (
                    "No active report is "
                    "available for question "
                    "answering."
                )

                state["current_step"] = (
                    CurrentStep.DONE.value
                )

                return state

            # Validate user question

            if not report_chat_query.strip():

                logger.warning(
                    "Empty report chat query",
                )

                state[
                    "report_chat_response"
                ] = (
                    "Please provide a valid "
                    "question about the report."
                )

                state["current_step"] = (
                    CurrentStep.DONE.value
                )

                return state

            logger.info(
                "Using compressed workspace "
                "memory for report chat",
            )
            
            print(
                "\n===== COMPRESSED REPORT CONTEXT =====\n"
            )

            print(
                state.get(
                    "compressed_report_context",
                    ""
                )
            )

            response = (
                ReportChatTools
                .answer_report_question(
                    report=(
                        report_context
                    ),
                    question=(
                        report_chat_query
                    ),
                )
            )

            state[
                "report_chat_response"
            ] = response

            # Update conversational memory

            MemoryManager.update_memory(
                state=state,
                user_query=(
                    report_chat_query
                ),
                assistant_response=(
                    response
                ),
            )

            state["current_step"] = (
                CurrentStep.DONE.value
            )

            logger.info(
                "Report chat completed",
            )

            return state

        except Exception as error:

            logger.error(
                "Report chat failed: %s",
                str(error),
            )

            state["error"] = str(
                error,
            )

            state["current_step"] = (
                CurrentStep.ERROR.value
            )

            return state