from langsmith import traceable

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


logger = setup_logger(__name__)


class ReportGenerationAgent:
    """Agent responsible for report generation."""

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

            summary = (
                ReportTools.generate_summary(
                    query=state["query"],
                    findings=(
                        state["key_findings"]
                    ),
                )
            )

            report = (
                ReportTools.format_report(
                    query=state["query"],
                    summary=summary,
                    findings=(
                        state["key_findings"]
                    ),
                    analysis=(
                        state[
                            "analysis_summary"
                        ]
                    ),
                    citations=(
                        state["citations"]
                    ),
                )
            )

            state["final_report"] = (
                report
            )

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

            state["error"] = str(error)

            state["current_step"] = (
                CurrentStep.ERROR.value
            )

            return state