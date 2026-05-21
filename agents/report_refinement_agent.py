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

from tools.report_compression_tools import (
    ReportCompressionTools,
)

from tools.report_refinement_tools import (
    ReportRefinementTools,
)

from tools.report_section_tools import (
    ReportSectionTools,
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
    """Agent responsible for intelligent section-aware refinement."""

    @staticmethod
    def _update_workspace_state(
        state: ResearchState,
        reconstructed_report: str,
        report_sections: dict[str, str],
        refinement_query: str,
        report_section_order: list[str],
        updated_section_name: str,
    ) -> None:
        """
        Update workspace state after refinement.
        """

        # Update full report

        state[
            "active_report"
        ] = reconstructed_report

        # Update structured sections

        state[
            "report_sections"
        ] = report_sections

        state[
            "report_section_order"
        ] = report_section_order

        # Incrementally update
        # compressed workspace memory

        updated_section_content = (
            report_sections.get(
                updated_section_name,
                "",
            )
        )

        previous_context = (
            state.get(
                "compressed_report_context",
                "",
            )
        )

        compressed_context = (
            ReportCompressionTools
            .update_compressed_context(
                previous_context=(
                    previous_context
                ),

                updated_section_name=(
                    updated_section_name
                ),

                updated_section_content=(
                    updated_section_content
                ),

                refinement_query=(
                    refinement_query
                ),
            )
        )

        state[
            "compressed_report_context"
        ] = compressed_context

        # Store report history

        ReportHistoryUtility.add_report(
            state=state,
            query=(
                state["query"]
            ),
            report=(
                reconstructed_report
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
                    reconstructed_report
                ),

                "mode": (
                    "REPORT_REFINEMENT"
                ),

                "refinement_query": (
                    refinement_query
                ),
            }
        )
        
    @staticmethod
    @traceable(
        name="report_refinement_agent",
    )
    def run(
        state: ResearchState,
    ) -> ResearchState:

        try:

            logger.info(
                "Starting intelligent "
                "report refinement",
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

            report_sections = (
                state.get(
                    "report_sections",
                    {},
                )
            )
            
            report_section_order = (
                state.get(
                    "report_section_order",
                    [],
                )
            )

            # Validate report

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

            # Validate query

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

            # Extract sections if missing
            # Use persisted workspace sections

            if (
                not report_sections
                or not report_section_order
            ):

                logger.info(
                    "Extracting initial "
                    "report sections",
                )

                extracted_data = (
                    ReportSectionTools
                    .extract_sections(
                        report=(
                            active_report
                        ),
                    )
                )

                report_sections = (
                    extracted_data[
                        "sections"
                    ]
                )

                report_section_order = (
                    extracted_data[
                        "section_order"
                    ]
                )

            else:

                logger.info(
                    "Using persisted "
                    "workspace sections",
                )

            # Intent classification

            logger.info(
                "Classifying refinement intent",
            )

            refinement_intent = (
                ReportRefinementTools
                .classify_refinement_intent(
                    refinement_query=(
                        refinement_query
                    ),
                    existing_sections=(
                        list(
                            report_sections
                            .keys()
                        )
                    ),
                )
            )

            intent = (
                refinement_intent.get(
                    "intent",
                    "modify_section",
                )
            )

            target_section = (
                refinement_intent.get(
                    "target_section",
                    "analysis",
                )
            )

            target_section = (
                ReportSectionTools
                .normalize_section_name(
                    target_section
                )
            )
            
            print(
                "\n===== FINAL REPORT SECTIONS ====="
            )

            for key in report_sections:
                print(key)

            logger.info(
                "Refinement intent: %s",
                intent,
            )

            logger.info(
                "Target section: %s",
                target_section,
            )

            # MODIFY EXISTING SECTION

            if (
                intent
                == "modify_section"
            ):

                existing_content = (
                    report_sections.get(
                        target_section,
                        "",
                    )
                )

                updated_section = (
                    ReportRefinementTools
                    .refine_section(
                        section_name=(
                            target_section
                        ),
                        section_content=(
                            existing_content
                        ),
                        refinement_query=(
                            refinement_query
                        ),
                    )
                )

                report_sections[
                    target_section
                ] = updated_section

            # ADD NEW SECTION

            elif (
                intent
                == "add_section"
            ):

                logger.info(
                    "Generating new section",
                )

                new_section = (
                    ReportRefinementTools
                    .generate_new_section(
                        section_name=(
                            target_section
                        ),

                        report_topic=(
                            state["query"]
                        ),

                        refinement_query=(
                            refinement_query
                        ),

                        existing_sections=(
                            report_section_order
                        ),
                    )
                )
                
                print(
                    "\n===== GENERATED NEW SECTION ====="
                )

                print(new_section)

                report_sections[
                    target_section
                ] = new_section
                
                placement_decision = (
                    ReportRefinementTools
                    .determine_section_placement(
                        new_section=(
                            target_section
                        ),
                        existing_sections=(
                            report_section_order
                        ),
                    )
                )
                
                insert_before = (
                    placement_decision.get(
                        "insert_before"
                    )
                )

                insert_after = (
                    placement_decision.get(
                        "insert_after"
                    )
                )

                if (
                    insert_before
                    and insert_before
                    in report_section_order
                ):

                    index = (
                        report_section_order
                        .index(
                            insert_before
                        )
                    )

                    report_section_order.insert(
                        index,
                        target_section,
                    )

                elif (
                    insert_after
                    and insert_after
                    in report_section_order
                ):

                    index = (
                        report_section_order
                        .index(
                            insert_after
                        )
                    )

                    report_section_order.insert(
                        index + 1,
                        target_section,
                    )

                else:

                    report_section_order.append(
                        target_section
                    )
                                                
                print(
                    "\n===== UPDATED REPORT SECTIONS ====="
                )

                for key in report_sections:
                    print(key)

            # Reconstruct report

            logger.info(
                "Reconstructing report",
            )

            reconstructed_report = (
                ReportSectionTools
                .reconstruct_report(
                    sections=(
                        report_sections
                    ),
                    section_order=(
                        report_section_order
                    ),
                )
            )

            # Update workspace

            (
                ReportRefinementAgent
                ._update_workspace_state(
                    state=state,

                    reconstructed_report=(
                        reconstructed_report
                    ),

                    report_sections=(
                        report_sections
                    ),

                    report_section_order=(
                        report_section_order
                    ),

                    refinement_query=(
                        refinement_query
                    ),

                    updated_section_name=(
                        target_section
                    ),
                )
            )
            
            # Update memory

            MemoryManager.update_memory(
                state=state,
                user_query=(
                    refinement_query
                ),
                assistant_response=(
                    reconstructed_report
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