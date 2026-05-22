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

    MAX_SUMMARY_LENGTH = 4000
    
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

        # Defensive normalization

        if not isinstance(
            title,
            str,
        ):

            title = str(title)

        if not isinstance(
            abstract,
            str,
        ):

            abstract = str(
                abstract
            )

        return {
            "title": title.strip(),
            "abstract": (
                abstract.strip()
            ),
        }

    @staticmethod
    def generate_summary(
        query: str,
        findings: list[str],
    ) -> str:
        """Generate executive summary."""

        if not findings:

            return (
                "No sufficient findings "
                "were available to generate "
                "a research summary."
            )

        llm = (
            LLMFactory
            .create_qwen_llm(
                temperature=0.2,
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
                            "research summarization "
                            "agent.\n\n"

                            "Generate a concise and "
                            "professional executive "
                            "summary.\n\n"

                            "Rules:\n"
                            "- preserve factual accuracy\n"
                            "- avoid hallucinations\n"
                            "- focus on important findings\n"
                            "- summarize technical insights\n"
                            "- keep the summary concise\n"
                            "- return only valid JSON\n\n"

                            "Format:\n"
                            "{{\n"
                            '  "summary": "..."\n'
                            "}}"
                        ),
                    ),
                    (
                        "human",
                        (
                            "Research Query:\n"
                            "{query}\n\n"
                            "Research Findings:\n"
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
                content=(
                    response.content
                ),
                fallback={
                    "summary": findings_text,
                },
            )
        )

        summary = (
            parsed_response.get(
                "summary",
                findings_text,
            )
        )

        # Defensive normalization

        if not isinstance(
            summary,
            str,
        ):

            summary = str(
                summary
            )

        return (
            summary.strip()[
                :ReportTools
                .MAX_SUMMARY_LENGTH
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
                temperature=0.2,
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
                            "- maintain professional "
                            "report structure\n"
                            "- preserve factual accuracy\n"
                            "- avoid hallucinations\n"
                            "- avoid removing valuable "
                            "existing sections\n"
                            "- keep the report coherent\n\n"

                            "The refinement may request:\n"
                            "- additional sections\n"
                            "- updated information\n"
                            "- more technical depth\n"
                            "- simplification\n"
                            "- practical applications\n"
                            "- better conclusions\n\n"

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

        return (
            parsed_response.get(
                "refined_report",
                existing_report,
            )
        )

    @staticmethod
    def format_report(
        title: str,
        query: str,
        abstract: str,
        summary: str,
        findings: list[str],
        analysis: str,
        citations: list[Citation],
    ) -> str:
        """Format professional research report."""

        findings_text = "\n".join(
            [
                f"- {finding}"
                for finding in findings
            ]
        )

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

        return f"""
# Research Report

## Title

{title}

## Abstract

{abstract}

## Introduction

This report provides a structured research analysis on:

"{query}"

The research examines major developments, technical foundations, practical applications, emerging trends, and critical insights associated with the topic.

The analysis is based on information retrieved from multiple research sources, including web-based resources, academic references, and contextual analytical findings.

The objective of this report is to synthesize relevant evidence into a professional and coherent research document that highlights key findings, technical observations, and broader implications related to the subject area.

## Executive Summary

{summary}

## Background & Context

This report explores the topic:
"{query}"

The analysis focuses on retrieved research findings, technical insights, practical implications, and relevant evidence gathered from multiple sources.

## Key Findings

{findings_text}

## Analysis & Insights

{analysis}

## Conclusion

The research findings provide important insights into the topic:
"{query}"

The report highlights key technical, practical, and conceptual observations derived from the analyzed sources.

While the retrieved evidence provides useful understanding of the topic, additional research may further improve coverage and depth in rapidly evolving areas.

## References

{citations_text}
"""