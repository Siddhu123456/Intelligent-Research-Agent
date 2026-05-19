from langchain_core.prompts import (
    ChatPromptTemplate,
)

from state.models import Citation
from utils.llm_factory import (
    LLMFactory,
)


class ReportTools:
    """Tools for report generation."""

    @staticmethod
    def generate_summary(
        query: str,
        findings: list[str],
    ) -> str:
        llm = (
            LLMFactory.create_qwen_llm(
                temperature=0.2,
            )
        )

        findings_text = "\n".join(
            findings,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an expert "
                        "research summarizer."
                    ),
                ),
                (
                    "human",
                    (
                        "Query:\n{query}\n\n"
                        "Findings:\n{findings}"
                    ),
                ),
            ]
        )

        chain = prompt | llm

        response = chain.invoke(
            {
                "query": query,
                "findings": findings_text,
            }
        )

        return response.content

    @staticmethod
    def format_report(
        query: str,
        summary: str,
        findings: list[str],
        analysis: str,
        citations: list[Citation],
    ) -> str:
        findings_text = "\n".join(
            [
                f"- {finding}"
                for finding in findings
            ]
        )

        citations_text = "\n".join(
            [
                (
                    f"- Claim: "
                    f"{citation.claim}\n"
                    f"  Source: "
                    f"{citation.source}\n"
                    f"  Document ID: "
                    f"{citation.doc_id}"
                )
                for citation in citations
            ]
        )

        return f"""
# Research Report

## Query
{query}

## Executive Summary
{summary}

## Key Findings
{findings_text}

## Detailed Analysis
{analysis}

## Sources
{citations_text}
"""