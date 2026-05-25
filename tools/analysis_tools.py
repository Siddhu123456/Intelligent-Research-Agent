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

from tools.prompts.analysis_tools_prompts import (
    ANALYSIS_EXTRACT_FINDINGS_SYSTEM_PROMPT,
    ANALYSIS_GENERATE_SUMMARY_SYSTEM_PROMPT,
    ANALYSIS_IDENTIFY_CONTRADICTIONS_SYSTEM_PROMPT,
)


class AnalysisTools:
    """Tools for research analysis."""

    MAX_FINDINGS = 5

    CONTEXT_DOCUMENT_LIMIT = 5

    CONTEXT_CONTENT_LIMIT = 1000

    @staticmethod
    def extract_key_findings(
        documents: list[Document],
        query: str,
    ) -> list[str]:
        """Extract important findings."""

        if not documents:

            return []

        llm = (
            LLMFactory
            .create_qwen_llm(
                temperature=0.1,
            )
        )

        limited_documents = (
            documents[
                :AnalysisTools
                .CONTEXT_DOCUMENT_LIMIT
            ]
        )

        context = "\n\n".join(
            [
                (
                    f"Title: "
                    f"{document.title}\n"
                    f"Content: "
                    f"{document.content[
                        :AnalysisTools
                        .CONTEXT_CONTENT_LIMIT
                    ]}"
                )
                for document in (
                    limited_documents
                )
            ]
        )

        prompt = (
            ChatPromptTemplate
            .from_messages(
                [
                    (
                        "system",
                        ANALYSIS_EXTRACT_FINDINGS_SYSTEM_PROMPT,
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

        parsed_response = (
            JSONParser.safe_extract(
                content=(
                    response.content
                ),
                fallback={
                    "findings": [],
                },
            )
        )

        findings = (
            parsed_response.get(
                "findings",
                [],
            )
        )

        cleaned_findings = [
            finding.strip()
            for finding in findings
            if finding.strip()
        ]

        return cleaned_findings[
            :AnalysisTools
            .MAX_FINDINGS
        ]

    @staticmethod
    def generate_analysis_summary(
        findings: list[str],
    ) -> str:
        """Generate analysis summary."""

        if not findings:

            return (
                "Insufficient findings "
                "available for analysis."
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
                        ANALYSIS_GENERATE_SUMMARY_SYSTEM_PROMPT,
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

        return (
            parsed_response.get(
                "summary",
                findings_text,
            )
            .strip()
        )

    @staticmethod
    def identify_contradictions(
        findings: list[str],
    ) -> str:
        """Identify contradictions and limitations."""

        if not findings:

            return (
                "No findings available "
                "for contradiction analysis."
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
                        ANALYSIS_IDENTIFY_CONTRADICTIONS_SYSTEM_PROMPT,
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

        parsed_response = (
            JSONParser.safe_extract(
                content=(
                    response.content
                ),
                fallback={
                    "analysis":
                    (
                        "No major contradictions "
                        "detected."
                    ),
                },
            )
        )

        return (
            parsed_response.get(
                "analysis",
                (
                    "No major contradictions "
                    "detected."
                ),
            )
            .strip()
        )

    @staticmethod
    def score_confidence(
        documents: list[Document],
    ) -> float:
        """Calculate retrieval confidence."""

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
        """Build citations from findings."""

        citations = []

        usable_documents = (
            documents[
                : len(findings)
            ]
        )

        for (
            finding,
            document,
        ) in zip(
            findings,
            usable_documents,
        ):

            citations.append(
                Citation(
                    doc_id=document.id,
                    claim=finding,
                    source=document.title,
                    url=document.url,
                )
            )

        return citations