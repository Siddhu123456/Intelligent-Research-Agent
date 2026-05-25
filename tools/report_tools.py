from langchain_core.prompts import (
    ChatPromptTemplate,
)

from state.models import (
    Citation,
)

from utils.json_parser import (
    JSONParser,
)

from utils.llm_factory import (
    LLMFactory,
)

from tools.prompts.report_tools_prompts import (
    REPORT_BODY_SYSTEM_PROMPT,
    REPORT_METADATA_SYSTEM_PROMPT,
    REPORT_REFINE_EXISTING_REPORT_SYSTEM_PROMPT,
)


class ReportTools:
    """Tools for report generation."""

    MAX_REPORT_BODY_LENGTH = 12000

    @staticmethod
    def generate_report_metadata(
        query: str,
        findings: list[str],
    ) -> dict:
        """
        Generate professional
        report title and abstract.
        """

        llm = (
            LLMFactory
            .create_qwen_llm(
                temperature=0.2,
            )
        )

        findings_text = "\n".join(
            findings[:5]
        )

        prompt = (
            ChatPromptTemplate
            .from_messages(
                [
                    (
                        "system",
                        REPORT_METADATA_SYSTEM_PROMPT,
                    ),
                    (
                        "human",
                        (
                            "Research Topic:\n"
                            "{query}\n\n"

                            "Key Findings:\n"
                            "{findings}"
                        ),
                    ),
                ]
            )
        )

        chain = (
            prompt
            | llm
        )

        response = chain.invoke(
            {
                "query": query,
                "findings": findings_text,
            }
        )

        parsed_response = (
            JSONParser.safe_extract(
                content=response.content,
                fallback={
                    "title": query,
                    "abstract": findings_text,
                },
            )
        )

        title = (
            parsed_response.get(
                "title",
                query,
            )
        )

        abstract = (
            parsed_response.get(
                "abstract",
                findings_text,
            )
        )

        if not isinstance(
            title,
            str,
        ):

            title = str(
                title
            )

        if not isinstance(
            abstract,
            str,
        ):

            abstract = str(
                abstract
            )

        return {
            "title": (
                title.strip()
            ),
            "abstract": (
                abstract.strip()
            ),
        }

    @staticmethod
    def generate_report_body(
        query: str,
        findings: list[str],
        analysis: str,
    ) -> str:
        """
        Generate fully dynamic
        research report body.
        """

        if not findings:

            return (
                "No sufficient findings "
                "were available to generate "
                "a research report."
            )

        llm = (
            LLMFactory
            .create_qwen_llm(
                temperature=0.1,
            )
        )

        findings_text = "\n".join(
            findings,
        )

        prompt = (
            ChatPromptTemplate
            .from_messages(
                [
                    (
                        "system",
                        REPORT_BODY_SYSTEM_PROMPT,
                    ),
                    (
                        "human",
                        (
                            "Research Topic:\n"
                            "{query}\n\n"

                            "Research Findings:\n"
                            "{findings}\n\n"

                            "Analysis:\n"
                            "{analysis}"
                        ),
                    ),
                ]
            )
        )

        chain = (
            prompt
            | llm
        )

        response = chain.invoke(
            {
                "query": query,
                "findings": findings_text,
                "analysis": analysis,
            }
        )

        parsed_response = (
            JSONParser.safe_extract(
                content=(
                    response.content
                ),
                fallback={
                    "report_body": (
                        findings_text
                    ),
                },
            )
        )

        report_body = (
            parsed_response.get(
                "report_body",
                findings_text,
            )
        )

        if not isinstance(
            report_body,
            str,
        ):

            report_body = str(
                report_body
            )

        return (
            report_body.strip()[
                :ReportTools
                .MAX_REPORT_BODY_LENGTH
            ]
        )

    @staticmethod
    def refine_existing_report(
        existing_report: str,
        refinement_query: str,
    ) -> str:
        """Refine existing report."""

        llm = (
            LLMFactory
            .create_qwen_llm(
                temperature=0.1,
            )
        )

        prompt = (
            ChatPromptTemplate
            .from_messages(
                [
                    (
                        "system",
                        REPORT_REFINE_EXISTING_REPORT_SYSTEM_PROMPT,
                    ),
                    (
                        "human",
                        (
                            "Existing Report:\n"
                            "{existing_report}\n\n"

                            "Refinement Request:\n"
                            "{refinement_query}"
                        ),
                    ),
                ]
            )
        )

        chain = (
            prompt
            | llm
        )

        response = chain.invoke(
            {
                "existing_report": (
                    existing_report
                ),
                "refinement_query": (
                    refinement_query
                ),
            }
        )

        parsed_response = (
            JSONParser.safe_extract(
                content=(
                    response.content
                ),
                fallback={
                    "refined_report": (
                        existing_report
                    ),
                },
            )
        )

        refined_report = (
            parsed_response.get(
                "refined_report",
                existing_report,
            )
        )

        if not isinstance(
            refined_report,
            str,
        ):

            refined_report = str(
                refined_report
            )

        return refined_report.strip()

    @staticmethod
    def format_report(
        title: str,
        query: str,
        abstract: str,
        report_body: str,
        citations: list[Citation],
    ) -> str:
        """
        Format flexible
        professional report.
        """

        seen_urls = set()

        formatted_references = []

        reference_index = 1

        for citation in citations:

            if (
                not citation.url
            ):

                continue

            if (
                citation.url
                in seen_urls
            ):

                continue

            seen_urls.add(
                citation.url
            )

            formatted_references.append(
                (
                    f"{reference_index}. "
                    f"{citation.source}\n"
                    f"URL: {citation.url}"
                )
            )

            reference_index += 1

        citations_text = "\n\n".join(
            formatted_references
        )

        report = f"""
# {title}

## Abstract

{abstract}

## Introduction

This report presents a detailed research analysis on:

"{query}"

The objective of this research is to explore the topic comprehensively using retrieved findings, technical analysis, practical insights, and supporting evidence gathered from multiple sources.

{report_body}
"""

        if citations_text.strip():

            report += f"""

## References

{citations_text}
"""

        return report.strip()