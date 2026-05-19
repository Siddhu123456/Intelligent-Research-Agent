from langsmith import (
    traceable,
)

from state.constants import (
    CurrentStep,
)

from state.schema import (
    ResearchState,
)

from tools.document_tools import (
    DocumentTools,
)

from utils.logger import (
    setup_logger,
)


logger = setup_logger(
    __name__,
)


class DocumentGenerationAgent:
    """Agent responsible for document generation."""

    @staticmethod
    @traceable(
        name="document_generation_agent",
    )
    def run(
        state: ResearchState,
    ) -> ResearchState:

        try:

            logger.info(
                "Starting document generation",
            )

            active_report = (
                state.get(
                    "active_report",
                    "",
                )
            )

            if not active_report.strip():

                logger.warning(
                    "No active report available "
                    "for document generation",
                )

                state["error"] = (
                    "No active report available "
                    "for PDF generation."
                )

                state["current_step"] = (
                    CurrentStep.ERROR.value
                )

                return state

            pdf_bytes = (
                DocumentTools
                .generate_pdf(
                    report=active_report,
                )
            )

            state[
                "generated_pdf"
            ] = pdf_bytes

            state["current_step"] = (
                CurrentStep.DONE.value
            )

            logger.info(
                "Document generation completed",
            )

            return state

        except Exception as error:

            logger.error(
                "Document generation failed: %s",
                str(error),
            )

            state["error"] = str(
                error,
            )

            state["current_step"] = (
                CurrentStep.ERROR.value
            )

            return state