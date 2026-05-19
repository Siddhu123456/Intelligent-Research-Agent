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

from tools.report_tools import (
    ReportTools,
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


class ReportGenerationAgent:
    """Agent responsible for report generation."""

    REPORT_REFINEMENT_MODE = (
        "REPORT_REFINEMENT"
    )

    @staticmethod
    def _store_report_version(
        state: ResearchState,
        report: str,
    ) -> None:
        """Store report version."""

        state[
            "report_version_history"
        ].append(
            {
                "query": (
                    state["query"]
                ),
                "report": report,
                "mode": (
                    state["mode"]
                ),
                "refinement_query": (
                    state[
                        "refinement_query"
                    ]
                ),
            }
        )

    @staticmethod
    def _generate_new_report(
        state: ResearchState,
    ) -> str:
        """Generate fresh report."""

        summary = (
            ReportTools
            .generate_summary(
                query=(
                    state["query"]
                ),
                findings=(
                    state[
                        "key_findings"
                    ]
                ),
            )
        )

        return (
            ReportTools
            .format_report(
                query=(
                    state["query"]
                ),
                summary=summary,
                findings=(
                    state[
                        "key_findings"
                    ]
                ),
                analysis=(
                    state[
                        "analysis_summary"
                    ]
                ),
                citations=(
                    state[
                        "citations"
                    ]
                ),
            )
        )

    @staticmethod
    def _refine_existing_report(
        state: ResearchState,
    ) -> str:
        """Refine active workspace report."""

        existing_report = (
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

        if not existing_report:

            logger.warning(
                "No active report found. "
                "Falling back to fresh "
                "report generation.",
            )

            return (
                ReportGenerationAgent
                ._generate_new_report(
                    state,
                )
            )

        logger.info(
            "Applying report refinement",
        )

        return (
            ReportTools
            .refine_existing_report(
                existing_report=(
                    existing_report
                ),
                refinement_query=(
                    refinement_query
                ),
            )
        )

    @staticmethod
    @traceable(
        name="report_generation_agent",
    )
    def run(
        state: ResearchState,
    ) -> ResearchState:

        try:

            logger.info(
                "Starting report generation",
            )

            mode = state.get(
                "mode",
                "REPORT_GENERATION",
            )

            # Refinement workflow

            if (
                mode
                == (
                    ReportGenerationAgent
                    .REPORT_REFINEMENT_MODE
                )
            ):

                report = (
                    ReportGenerationAgent
                    ._refine_existing_report(
                        state,
                    )
                )

            # Fresh report workflow

            else:

                report = (
                    ReportGenerationAgent
                    ._generate_new_report(
                        state,
                    )
                )

            # Update active workspace report

            state[
                "active_report"
            ] = report

            # Store report history

            ReportHistoryUtility.add_report(
                state=state,
                query=(
                    state["query"]
                ),
                report=report,
            )

            # Store report versions

            (
                ReportGenerationAgent
                ._store_report_version(
                    state=state,
                    report=report,
                )
            )

            # Update conversational memory

            MemoryManager.update_memory(
                state=state,
                user_query=(
                    state["query"]
                ),
                assistant_response=(
                    report
                ),
            )

            # Mark workflow completed

            state["current_step"] = (
                CurrentStep.DONE.value
            )

            logger.info(
                "Report generation completed",
            )

            return state

        except Exception as error:

            logger.error(
                "Report generation failed: %s",
                str(error),
            )

            state["error"] = str(
                error,
            )

            state["current_step"] = (
                CurrentStep.ERROR.value
            )

            return state