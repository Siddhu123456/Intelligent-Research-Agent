from langsmith import traceable

from state.constants import CurrentStep
from state.schema import ResearchState
from tools.analysis_tools import (
    AnalysisTools,
)
from utils.logger import setup_logger


logger = setup_logger(__name__)


class AnalysisAgent:
    """Agent responsible for research analysis."""

    @staticmethod
    @traceable(
        name="analysis_agent",
    )
    def run(
        state: ResearchState,
    ) -> ResearchState:
        try:
            logger.info(
                "Starting analysis process",
            )

            documents = (
                state[
                    "retrieved_documents"
                ]
            )

            findings = (
                AnalysisTools.extract_key_findings(
                    documents=documents,
                    query=state["query"],
                )
            )

            summary = (
                AnalysisTools
                .generate_analysis_summary(
                    findings,
                )
            )

            contradictions = (
                AnalysisTools
                .identify_contradictions(
                    findings,
                )
            )

            citations = (
                AnalysisTools.build_citations(
                    findings=findings,
                    documents=documents,
                )
            )

            confidence_score = (
                AnalysisTools.score_confidence(
                    documents,
                )
            )

            state["key_findings"] = (
                findings
            )

            state["analysis_summary"] = (
                f"{summary}\n\n"
                f"Contradictions & "
                f"Limitations:\n"
                f"{contradictions}"
            )

            state["citations"] = (
                citations
            )

            state["low_confidence"] = (
                confidence_score < 0.6
            )

            state["current_step"] = (
                CurrentStep.ANALYZED.value
            )

            logger.info(
                (
                    "Analysis completed "
                    "with confidence %.2f"
                ),
                confidence_score,
            )

            return state

        except Exception as error:
            logger.error(
                "Analysis failed: %s",
                str(error),
            )

            state["error"] = str(error)

            state["current_step"] = (
                CurrentStep.ERROR.value
            )

            return state