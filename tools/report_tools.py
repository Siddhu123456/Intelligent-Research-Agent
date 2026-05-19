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

        prompt = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        (
                            "You are an expert research summarization "
                            "agent.\n\n"

                            "Your task is to generate a concise and "
                            "professional research summary.\n\n"

                            "IMPORTANT RULES:\n"
                            "- Prioritize information directly relevant "
                            "to the user's query.\n"
                            "- Ignore unrelated findings.\n"
                            "- Do not hallucinate unsupported claims.\n"
                            "- Keep the summary factual and concise.\n\n"

                            "FALLBACK BEHAVIOR:\n"
                            "- If the provided findings are mostly "
                            "irrelevant or insufficient, generate a "
                            "clear high-level summary of the topic "
                            "based on general knowledge.\n"
                            "- In fallback mode, explain the topic "
                            "clearly without referencing unrelated findings.\n\n"

                            "Focus on:\n"
                            "- key concepts\n"
                            "- technical insights\n"
                            "- major applications\n"
                            "- important conclusions\n\n"

                            "Return ONLY valid JSON.\n\n"

                            "Format:\n"
                            "{{\n"
                            '  "summary": "..." \n'
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

        try:

            parsed_response = (
                JSONParser.extract_json(
                    response.content,
                )
            )

            return (
                parsed_response[
                    "summary"
                ]
                .strip()[:4000]
            )

        except Exception:

            return findings_text[:4000]

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