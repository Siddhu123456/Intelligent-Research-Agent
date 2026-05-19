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

from tools.report_refinement_tools import (
    ReportRefinementTools,
)

from utils.logger import (
    setup_logger,
)

from utils.report_history import (
    ReportHistoryUtility,
)


logger = setup_logger(
    __name__,
)


class ReportRefinementAgent:
    """Agent responsible for intelligent report refinement."""

    @staticmethod
    @traceable(
        name="report_refinement_agent",
    )
    def run(
        state: ResearchState,
    ) -> ResearchState:

        try:

            logger.info(
                "Starting report refinement",
            )

            active_report = (
                state.get(
                    "active_report",
                    "",
                )
            )

            refinement_query = (
                state.get(
                    "refinement_query",
                    "",
                )
            )

            if not active_report.strip():

                logger.warning(
                    "No active report available",
                )

                state["error"] = (
                    "No active report available "
                    "for refinement."
                )

                state["current_step"] = (
                    CurrentStep.ERROR.value
                )

                return state

            if not refinement_query.strip():

                logger.warning(
                    "Empty refinement query",
                )

                state["error"] = (
                    "Please provide a valid "
                    "refinement instruction."
                )

                state["current_step"] = (
                    CurrentStep.ERROR.value
                )

                return state

            refined_report = (
                ReportRefinementTools
                .refine_report(
                    report=(
                        active_report
                    ),
                    refinement_instruction=(
                        refinement_query
                    ),
                )
            )

            # Update active report

            state[
                "active_report"
            ] = refined_report

            # Store report history

            ReportHistoryUtility.add_report(
                state=state,
                query=(
                    state["query"]
                ),
                report=(
                    refined_report
                ),
            )

            # Store version history

            state[
                "report_version_history"
            ].append(
                {
                    "query": (
                        state["query"]
                    ),
                    "report": (
                        refined_report
                    ),
                    "mode": (
                        "REPORT_REFINEMENT"
                    ),
                    "refinement_query": (
                        refinement_query
                    ),
                }
            )

            # Update memory

            MemoryManager.update_memory(
                state=state,
                user_query=(
                    refinement_query
                ),
                assistant_response=(
                    refined_report
                ),
            )

            state["current_step"] = (
                CurrentStep.DONE.value
            )

            logger.info(
                "Report refinement completed",
            )

            return state

        except Exception as error:

            logger.error(
                "Report refinement failed: %s",
                str(error),
            )

            state["error"] = str(
                error,
            )

            state["current_step"] = (
                CurrentStep.ERROR.value
            )

            return state