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
                        (
                            "You are an expert "
                            "academic research writer.\n\n"

                            "Generate:\n"
                            "- a professional "
                            "research report title\n"
                            "- a concise abstract\n\n"

                            "IMPORTANT RULES:\n"
                            "- title should sound "
                            "professional\n"
                            "- do not over complicate the title\n"
                            "- do not include years in the title\n"
                            "- title should be concise\n"
                            "- abstract should summarize "
                            "the research\n"
                            "- avoid hallucinations\n"
                            "- preserve technical accuracy\n\n"

                            "Return ONLY valid JSON.\n\n"

                            "Format:\n"
                            "{{\n"
                            '  "title": "...",\n'
                            '  "abstract": "..."\n'
                            "}}"
                        ),
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
                        (
                            "You are an expert "
                            "research report writer.\n\n"

                            "Generate a professional "
                            "research report body.\n\n"

                            "IMPORTANT RULES:\n"
                            "- generate natural "
                            "section structures\n"
                            "- create sections dynamically\n"
                            "- structure the report using markdown headings\n"
                            "- generate multiple topic-specific sections\n"
                            "- ONLY use ## for major top-level sections\n"
                            "- use ### for ALL subsections\n"
                            "- NEVER generate nested ## headings\n"
                            "- maintain proper markdown hierarchy\n"
                            "- subsection headings MUST use ###\n"
                            "- avoid writing the entire report as one block\n"
                            "- separate technical concepts into distinct sections\n"
                            "- organize the report professionally\n"
                            "- create clear thematic segmentation\n"
                            "- include technical explanations "
                            "when relevant\n"
                            "- include practical applications "
                            "when relevant\n"
                            "- include risks and limitations "
                            "when relevant\n"
                            "- include future trends "
                            "when relevant\n"
                            "- preserve factual accuracy\n"
                            "- avoid hallucinations\n"
                            "- maintain professional tone\n"
                            "- do not generate references section\n"
                            "- do not generate title section\n"
                            "- do not generate abstract section\n"
                            "- introduction is already handled\n"
                            "- use clean markdown formatting\n"
                            "- create topic-specific sections\n"
                            "- avoid generic templates\n\n"

                            "The report structure should adapt "
                            "naturally to the topic.\n\n"
                            
                            "The report MUST contain:\n"
                            "- multiple markdown sections\n"
                            "- topic-specific headings\n"
                            "- clear content organization\n"
                            "- professional research structure\n"

                            "Avoid producing one continuous paragraph block.\n\n"
                            
                            "The markdown hierarchy MUST remain valid.\n"
                            "Do not create top-level sections inside other sections.\n"
                            "Nested topics must use ### headings.\n\n"
                            
                            "GOOD EXAMPLE:\n\n"

                            "## Neural Interfaces\n\n"

                            "Main section content.\n\n"

                            "### Flexible Electrodes\n\n"

                            "Subsection content.\n\n"

                            "### Signal Processing\n\n"

                            "Subsection content.\n\n"

                            "BAD EXAMPLE:\n\n"

                            "## Neural Interfaces\n\n"

                            "content...\n\n"

                            "## Flexible Electrodes\n\n"

                            "content...\n\n"

                            "Return ONLY valid JSON.\n\n"

                            "Format:\n"
                            "{{\n"
                            '  "report_body": "..."\n'
                            "}}"
                        ),
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
                        (
                            "You are an expert "
                            "research editor.\n\n"

                            "Your task is to refine "
                            "and improve an existing "
                            "research report based on "
                            "the user's refinement "
                            "instruction.\n\n"

                            "IMPORTANT RULES:\n"
                            "- preserve existing useful "
                            "content\n"
                            "- improve the report instead "
                            "of rewriting everything\n"
                            "- integrate the refinement "
                            "naturally\n"
                            "- maintain coherent organization\n"
                            "- allow natural restructuring "
                            "when useful\n"
                            "- preserve factual accuracy\n"
                            "- avoid hallucinations\n"
                            "- avoid removing valuable "
                            "existing sections\n"
                            "- keep the report coherent\n"
                            "- preserve markdown hierarchy\n"
                            "- ONLY use ## for top-level sections\n"
                            "- use ### for subsections\n"
                            "- NEVER generate nested ## headings\n"
                            "- preserve existing heading organization\n"
                            "- maintain proper markdown nesting\n\n"

                            "The refinement may request:\n"
                            "- additional sections\n"
                            "- updated information\n"
                            "- more technical depth\n"
                            "- simplification\n"
                            "- practical applications\n"
                            "- comparative analysis\n"
                            "- ethical discussions\n"
                            "- future trends\n\n"
                            
                            "GOOD EXAMPLE:\n\n"

                            "## Neural Interfaces\n\n"

                            "content...\n\n"

                            "### Flexible Electrodes\n\n"

                            "new subsection content\n\n"

                            "BAD EXAMPLE:\n\n"

                            "## Neural Interfaces\n\n"

                            "content...\n\n"

                            "## Flexible Electrodes\n\n"

                            "content...\n\n"

                            "Return ONLY valid JSON.\n\n"

                            "Format:\n"
                            "{{\n"
                            '  "refined_report": "..."\n'
                            "}}"
                        ),
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