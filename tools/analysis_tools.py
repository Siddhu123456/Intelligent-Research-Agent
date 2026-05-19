from langchain_core.prompts import (
    ChatPromptTemplate,
)

from state.models import (
    Citation,
    Document,
)

from utils.json_parser import (
    JSONParser,
)

from utils.llm_factory import (
    LLMFactory,
)


class AnalysisTools:
    """Tools for research analysis."""

    @staticmethod
    def extract_key_findings(
        documents: list[Document],
        query: str,
    ) -> list[str]:
        llm = (
            LLMFactory.create_qwen_llm(
                temperature=0.1,
            )
        )

        context = "\n\n".join(
            [
                (
                    f"Title: "
                    f"{document.title}\n"
                    f"Content: "
                    f"{document.content[:1000]}"
                )
                for document in documents
            ]
        )

        prompt = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        (
                            "You are an expert research analyst.\n\n"

                            "Analyze the retrieved documents and extract "
                            "up to 5 important findings relevant to the "
                            "user query.\n\n"

                            "IMPORTANT RULES:\n"
                            "- Prioritize findings directly related "
                            "to the query.\n"
                            "- Ignore irrelevant information.\n"
                            "- Do not hallucinate unsupported facts.\n"
                            "- Preserve factual accuracy.\n"
                            "- Avoid repetition.\n"
                            "- Keep findings concise and technical.\n"
                            "- If relevant information is limited, "
                            "extract the closest useful insights.\n\n"

                            "Focus on:\n"
                            "- technical findings\n"
                            "- important conclusions\n"
                            "- practical insights\n"
                            "- key observations\n\n"

                            "Return ONLY valid JSON.\n\n"

                            "Format:\n"
                            "{{\n"
                            '  "findings": [\n'
                            '    "...",\n'
                            '    "..."\n'
                            "  ]\n"
                            "}}"
                        ),
                    ),
                    (
                        "human",
                        (
                            "User Query:\n"
                            "{query}\n\n"
                            "Retrieved Context:\n"
                            "{context}"
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
                "context": context,
            }
        )

        try:

            parsed_response = (
                JSONParser.extract_json(
                    response.content,
                )
            )

            findings = (
                parsed_response.get(
                    "findings",
                    [],
                )
            )

            return findings[:5]

        except Exception:

            return []

    @staticmethod
    def generate_analysis_summary(
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
                            "You are an expert research analyst.\n\n"

                            "Generate a concise and professional "
                            "analysis summary.\n\n"

                            "IMPORTANT RULES:\n"
                            "- Prioritize findings relevant to the topic.\n"
                            "- Ignore unrelated observations.\n"
                            "- Do not hallucinate unsupported claims.\n"
                            "- Keep the summary factual and concise.\n\n"

                            "FALLBACK BEHAVIOR:\n"
                            "- If findings are weak, sparse, or partially "
                            "irrelevant, generate a clear high-level "
                            "summary of the overall topic using general "
                            "knowledge.\n"
                            "- Do not reference unrelated findings in "
                            "fallback mode.\n\n"

                            "Focus on:\n"
                            "- major insights\n"
                            "- implications\n"
                            "- technical conclusions\n"
                            "- practical observations\n\n"

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
                ].strip()
            )

        except Exception:

            return (
                findings_text
            )

    @staticmethod
    def identify_contradictions(
        findings: list[str],
    ) -> str:
        llm = (
            LLMFactory.create_qwen_llm(
                temperature=0.1,
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
                            "You are a contradiction analysis agent.\n\n"

                            "Analyze the findings for contradictions, "
                            "inconsistencies, uncertainty, and limitations.\n\n"

                            "IMPORTANT RULES:\n"
                            "- Only identify contradictions supported "
                            "by the findings.\n"
                            "- Do not invent conflicts.\n"
                            "- Ignore unrelated findings.\n"
                            "- Keep the analysis concise and factual.\n\n"

                            "FALLBACK BEHAVIOR:\n"
                            "- If no meaningful contradictions exist, "
                            "summarize the overall consistency and "
                            "limitations of the findings.\n\n"

                            "Focus on:\n"
                            "- conflicting statements\n"
                            "- uncertainty\n"
                            "- evidence limitations\n"
                            "- missing information\n\n"

                            "Return ONLY valid JSON.\n\n"

                            "Format:\n"
                            "{{\n"
                            '  "analysis": "..." \n'
                            "}}"
                        ),
                    ),
                    (
                        "human",
                        (
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
                    "analysis"
                ].strip()
            )

        except Exception:

            return (
                "No major contradictions detected."
            )

    @staticmethod
    def score_confidence(
        documents: list[Document],
    ) -> float:
        if not documents:
            return 0.0

        source_diversity = len(
            set(
                document.source
                for document in documents
            )
        )

        document_count = len(
            documents,
        )

        confidence = min(
            (
                (
                    document_count / 5
                )
                * 0.7
            )
            + (
                (
                    source_diversity / 3
                )
                * 0.3
            ),
            1.0,
        )

        return round(
            confidence,
            2,
        )

    @staticmethod
    def build_citations(
        findings: list[str],
        documents: list[Document],
    ) -> list[Citation]:
        citations: list[Citation] = []

        usable_documents = documents[
            : len(findings)
        ]

        for finding, document in zip(
            findings,
            usable_documents,
        ):
            citations.append(
                Citation(
                    doc_id=document.id,
                    claim=finding,
                    source=document.title,
                )
            )

        return citations