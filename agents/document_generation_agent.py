from langsmith import (
    traceable,
)

import os
import tempfile
from datetime import (
    datetime,
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

            def _safe_str(value):
                if value is None:
                    return ""
                if isinstance(value, str):
                    return value
                try:
                    return str(value)
                except Exception:
                    return ""

            def _safe_dict(value):
                if isinstance(value, dict):
                    return value
                if isinstance(value, str):
                    return {}
                if hasattr(value, "items"):
                    try:
                        return dict(value)
                    except Exception:
                        return {}
                return {}

            def _safe_list(value):
                if isinstance(value, list):
                    return value
                if isinstance(value, str):
                    return [value]
                if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                    try:
                        return list(value)
                    except Exception:
                        return []
                return []

            if not hasattr(state, "get"):
                logger.warning(
                    "DocumentGenerationAgent received invalid workflow state type %s; wrapping as active_report",
                    type(state).__name__,
                )
                state = {
                    "active_report": _safe_str(state),
                }
            elif not isinstance(state, dict):
                try:
                    state = dict(state)
                except Exception:
                    state = {
                        "active_report": _safe_str(state.get("active_report", "")) if hasattr(state, "get") else "",
                        "report_sections": _safe_dict(state.get("report_sections", {})) if hasattr(state, "get") else {},
                        "report_section_order": _safe_list(state.get("report_section_order", [])) if hasattr(state, "get") else [],
                        "citations": _safe_list(state.get("citations", [])) if hasattr(state, "get") else [],
                        "query": _safe_str(state.get("query", "")) if hasattr(state, "get") else "",
                        "report_title": _safe_str(state.get("report_title", "")) if hasattr(state, "get") else "",
                    }

            logger.info(
                "Starting document generation",
            )

            active_report = _safe_str(state.get("active_report", ""))

            if (
                not active_report
                or not active_report.strip()
            ):

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

            # Generate output path

            timestamp = (
                datetime.now()
                .strftime(
                    "%Y%m%d_%H%M%S"
                )
            )

            safe_title = (
                state.get(
                    "report_title",
                    "research_report",
                )
                .lower()
                .replace(
                    " ",
                    "_",
                )
                .replace(
                    "/",
                    "_",
                )
                .replace(
                    "\\",
                    "_",
                )
            )
            
            safe_title = "".join(
                char
                for char in safe_title
                if char.isalnum()
                or char in {
                    "_",
                    "-",
                }
            )

            safe_title = (
                safe_title[:80]
                or "research_report"
            )

            output_path = os.path.join(
                tempfile.gettempdir(),
                (
                    f"{safe_title}_"
                    f"{timestamp}.pdf"
                ),
            )

            # Build metadata

            metadata = {
                "topic": state.get(
                    "report_title",
                    state.get(
                        "query",
                        "Research Report",
                    ),
                ),
                "title": state.get(
                    "report_title",
                    state.get(
                        "query",
                        "Research Report",
                    ),
                ),
                "subtitle": "Generated by Intelligent Research Agent",
                "date": datetime.now().strftime("%Y-%m-%d"),
            }

            # Build sections

            sections = []

            report_sections = _safe_dict(state.get("report_sections", {}))
            report_order = _safe_list(state.get("report_section_order", []))

            # Prevent accidental title duplication.
            report_order = [
                section
                for section in report_order
                if isinstance(section, str)
                and section.lower() not in {
                    "title",
                    "report_title",
                }
            ]

            for section_name in report_order:
                content = report_sections.get(section_name, "")
                
                if (
                    not content
                    or not str(
                        content
                    ).strip()
                ):
                    continue

                formatted_title = (
                    section_name
                    .replace(
                        "_",
                        " ",
                    )
                    .title()
                )

                sections.append(
                    {
                        "title": (
                            formatted_title
                        ),

                        "content": (
                            content
                        ),
                    }
                )

            # Build references

            references = []

            citations = _safe_list(state.get("citations", []))

            for citation in citations:
                references.append(
                    {
                        "title": getattr(
                            citation,
                            "source",
                            "Unknown Source",
                        ),
                        "url": getattr(
                            citation,
                            "url",
                            "",
                        ),
                    }
                )
                
            # Fallback if sections
            # extraction failed.

            if not sections:

                sections.append(
                    {
                        "title": "Report",
                        "content": active_report,
                    }
                )

            # Generate PDF

            generated_path = (
                DocumentTools
                .generate_research_pdf(
                    output_path=(
                        output_path
                    ),

                    metadata=(
                        metadata
                    ),

                    sections=(
                        sections
                    ),

                    references=(
                        references
                    ),
                )
            )
            
            with open(
                generated_path,
                "rb",
            ) as pdf_file:
                logger.info(
                    "Loading generated PDF bytes",
                )

                state[
                    "generated_pdf"
                ] = pdf_file.read()
                
                if (
                    "run_metadata"
                    not in state
                    or not isinstance(
                        state[
                            "run_metadata"
                        ],
                        dict,
                    )
                ):

                    state[
                        "run_metadata"
                    ] = {}

                state[
                    "run_metadata"
                ][
                    "generated_pdf_filename"
                ] = os.path.basename(
                    generated_path
                )

            state["current_step"] = (
                CurrentStep.DONE.value
            )

            logger.info(
                "Document generation completed",
            )

            logger.info(
                "Generated PDF: %s",
                generated_path,
            )

            return state

        except Exception as error:

            logger.exception(
                "Document generation failed",
            )

            state["error"] = str(
                error,
            )

            state["current_step"] = (
                CurrentStep.ERROR.value
            )

            return state