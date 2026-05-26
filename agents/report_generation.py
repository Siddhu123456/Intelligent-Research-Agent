from utils.langsmith_wrapper import traceable

from datetime import (
    datetime,
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


from tools.report_section_tools import (
    ReportSectionTools,
)

from tools.report_vector_tools import (
    ReportVectorTools,
)

from utils.logger import (
    setup_logger,
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
                "title": (
                    state.get(
                        "report_title",
                        state.get(
                            "query",
                            "Research Report",
                        ),
                    )
                ),
                "query": (
                    state["query"]
                ),

                "report": report,

                "report_sections": (
                    state.get(
                        "report_sections",
                        {},
                    )
                ),

                "report_section_order": (
                    state.get(
                        "report_section_order",
                        [],
                    )
                ),

                "mode": (
                    "REPORT_GENERATION"
                ),

                "description": (
                    "Initial report generation"
                ),

                "timestamp": str(
                    datetime.now()
                ),
            }
        )

    @staticmethod
    def _generate_new_report(
        state: ResearchState,
    ) -> str:
        """Generate fresh report."""

        logger.info(
            "Generating report metadata",
        )

        metadata = (
            ReportTools
            .generate_report_metadata(
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

        report_title = (
            metadata["title"]
        )
        
        state[
            "report_title"
        ] = report_title

        abstract = (
            metadata["abstract"]
        )

        logger.info(
            "Generating dynamic report body",
        )

        report_body = (
            ReportTools
            .generate_report_body(
                query=(
                    state["query"]
                ),
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
            )
        )

        logger.info(
            "Formatting final report",
        )

        return (
            ReportTools
            .format_report(
                title=(
                    report_title
                ),
                query=(
                    state["query"]
                ),
                abstract=(
                    abstract
                ),
                report_body=(
                    report_body
                ),
                citations=(
                    state[
                        "citations"
                    ]
                ),
            )
        )

    # `_refine_existing_report` removed — report refinement is handled
    # by `ReportRefinementAgent` and `tools/report_refinement_tools.py`.

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

            # Generate fresh report (refinement handled elsewhere)

            report = (
                ReportGenerationAgent
                ._generate_new_report(
                    state,
                )
            )

            if not isinstance(
                report,
                str,
            ):

                report = str(
                    report
                )

            report = report.strip()
            
            # ── Update active workspace report ───────────

            state[
                "active_report"
            ] = report

            # ── Extract structured sections ──────────────

            logger.info(
                "Extracting report sections",
            )

            extracted_data = (
                ReportSectionTools
                .extract_sections(
                    report=report,
                )
            )

            state[
                "report_sections"
            ] = (
                extracted_data.get(
                    "sections",
                    {},
                )
            )

            state[
                "report_section_order"
            ] = (
                ReportSectionTools
                .get_section_order(
                    state[
                        "report_sections"
                    ]
                )
            )
            #  Store report embeddings 

            session_id = (
                state.get(
                    "session_id",
                    "",
                )
            )

            # Note: refinement-specific embedding clearing removed; the
            # dedicated refinement agent manages report embedding lifecycle.

            logger.info(
                "Storing report embeddings",
            )

            ReportVectorTools.store_report(
                sections=(
                    state[
                        "report_sections"
                    ]
                ),
                session_id=session_id,
            )
            
            # ── Compress report ONLY for
            # conversational memory / chat ───────────────

            logger.info(
                "Compressing report context",
            )

            # Compressed report summary removed; downstream consumers
            # should use semantic RAG over report sections instead.

            #  Store report versions 

            (
                ReportGenerationAgent
                ._store_report_version(
                    state=state,
                    report=report,
                )
            )

            #  Update conversational memory 

            logger.info(
                "Updating conversational memory",
            )

            MemoryManager.update_memory(
                state=state,
                user_query=(
                    state.get(
                        "refinement_query",
                        state["query"],
                    )
                ),
                assistant_response=(
                    report
                ),
            )

            #  Workflow completed 

            state[
                "current_step"
            ] = (
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

            state[
                "current_step"
            ] = (
                CurrentStep.ERROR.value
            )

            return state