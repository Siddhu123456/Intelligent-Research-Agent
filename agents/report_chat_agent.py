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

            if not active_report:

                logger.warning(
                    "No active report found",
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

            response = (
                ReportChatTools
                .answer_report_question(
                    report=(
                        active_report
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