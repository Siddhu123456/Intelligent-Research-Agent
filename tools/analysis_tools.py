from state.models import Citation
from state.models import Document
from utils.llm_factory import LLMFactory


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
                    f"Title: {document.title}\n"
                    f"Content: "
                    f"{document.content[:1000]}"
                )
                for document in documents
            ]
        )

        prompt = f"""
You are an expert research analyst.

User Query:
{query}

Retrieved Context:
{context}

Tasks:
1. Analyze the retrieved information.
2. Extract the 5 most important findings.
3. Ensure findings are factual.
4. Avoid repetition.

Return ONLY bullet points.
"""

        response = llm.invoke(
            prompt,
        )

        findings = [
            line.strip("- ").strip()
            for line in response.content.split(
                "\n",
            )
            if line.strip()
        ]

        return findings[:5]

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

        prompt = f"""
You are an expert research analyst.

Findings:
{findings_text}

Generate a concise research analysis
summarizing:
- major insights
- implications
- important observations

Keep it concise and professional.
"""

        response = llm.invoke(
            prompt,
        )

        return response.content

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

        prompt = f"""
Analyze the following findings.

Findings:
{findings_text}

Identify:
- contradictions
- inconsistencies
- uncertainty
- limitations

Return concise analysis.
"""

        response = llm.invoke(
            prompt,
        )

        return response.content

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