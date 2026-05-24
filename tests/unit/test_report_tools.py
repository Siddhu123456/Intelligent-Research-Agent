import pytest
from unittest.mock import Mock, patch

from tools.report_tools import (
    ReportTools,
)
from state.models import Citation


class TestReportTools:
    """Unit tests for ReportTools."""

    @patch(
        "tools.report_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_tools.JSONParser.safe_extract"
    )
    def test_generate_report_metadata_success(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test successful metadata generation."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"title": "AI Research", '
            '"abstract": '
            '"AI research abstract."}'
        )

        mock_chain = Mock()
        mock_chain.invoke.return_value = (
            mock_response
        )

        mock_prompt = Mock()

        mock_prompt.__or__ = Mock(
            return_value=mock_chain
        )

        mock_create_llm.return_value = (
            mock_llm
        )

        mock_safe_extract.return_value = {
            "title":
            "AI Research",
            "abstract":
            "AI research abstract.",
        }

        with patch(
            "tools.report_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            result = (
                ReportTools
                .generate_report_metadata(
                    query=(
                        "Artificial Intelligence"
                    ),
                    findings=[
                        "AI improves automation",
                    ],
                )
            )

        assert (
            result["title"]
            == "AI Research"
        )

        assert (
            result["abstract"]
            == (
                "AI research abstract."
            )
        )

    @patch(
        "tools.report_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_tools.JSONParser.safe_extract"
    )
    def test_generate_report_metadata_non_string(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test metadata normalization."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"title": 123, '
            '"abstract": 456}'
        )

        mock_chain = Mock()
        mock_chain.invoke.return_value = (
            mock_response
        )

        mock_prompt = Mock()

        mock_prompt.__or__ = Mock(
            return_value=mock_chain
        )

        mock_create_llm.return_value = (
            mock_llm
        )

        mock_safe_extract.return_value = {
            "title": 123,
            "abstract": 456,
        }

        with patch(
            "tools.report_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            result = (
                ReportTools
                .generate_report_metadata(
                    query="AI",
                    findings=["AI finding"],
                )
            )

        assert (
            result["title"]
            == "123"
        )

        assert (
            result["abstract"]
            == "456"
        )

    def test_generate_summary_empty_findings(
        self,
    ):
        """Test summary generation with empty findings."""

        result = (
            ReportTools
            .generate_summary(
                query="AI",
                findings=[],
            )
        )

        assert (
            result
            == (
                "No sufficient findings "
                "were available to generate "
                "a research summary."
            )
        )

    @patch(
        "tools.report_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_tools.JSONParser.safe_extract"
    )
    def test_generate_summary_success(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test successful summary generation."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"summary": '
            '"AI summary text."}'
        )

        mock_chain = Mock()
        mock_chain.invoke.return_value = (
            mock_response
        )

        mock_prompt = Mock()

        mock_prompt.__or__ = Mock(
            return_value=mock_chain
        )

        mock_create_llm.return_value = (
            mock_llm
        )

        mock_safe_extract.return_value = {
            "summary":
            "AI summary text."
        }

        with patch(
            "tools.report_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            result = (
                ReportTools
                .generate_summary(
                    query="AI",
                    findings=[
                        "Finding 1",
                        "Finding 2",
                    ],
                )
            )

        assert (
            result
            == "AI summary text."
        )

    @patch(
        "tools.report_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_tools.JSONParser.safe_extract"
    )
    def test_generate_summary_non_string(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test summary normalization."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"summary": 999}'
        )

        mock_chain = Mock()
        mock_chain.invoke.return_value = (
            mock_response
        )

        mock_prompt = Mock()

        mock_prompt.__or__ = Mock(
            return_value=mock_chain
        )

        mock_create_llm.return_value = (
            mock_llm
        )

        mock_safe_extract.return_value = {
            "summary": 999
        }

        with patch(
            "tools.report_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            result = (
                ReportTools
                .generate_summary(
                    query="AI",
                    findings=[
                        "Finding 1"
                    ],
                )
            )

        assert (
            result
            == "999"
        )

    @patch(
        "tools.report_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_tools.JSONParser.safe_extract"
    )
    def test_refine_existing_report_success(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test successful report refinement."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"refined_report": '
            '"Improved report content."}'
        )

        mock_chain = Mock()
        mock_chain.invoke.return_value = (
            mock_response
        )

        mock_prompt = Mock()

        mock_prompt.__or__ = Mock(
            return_value=mock_chain
        )

        mock_create_llm.return_value = (
            mock_llm
        )

        mock_safe_extract.return_value = {
            "refined_report":
            (
                "Improved report "
                "content."
            )
        }

        with patch(
            "tools.report_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            result = (
                ReportTools
                .refine_existing_report(
                    existing_report=(
                        "Old report"
                    ),
                    refinement_query=(
                        "Improve clarity"
                    ),
                )
            )

        assert (
            result
            == (
                "Improved report "
                "content."
            )
        )

    def test_format_report_success(
        self,
    ):
        """Test professional report formatting."""

        citations = [
            Citation(
                doc_id="doc_1",
                claim="AI improves efficiency",
                source="AI Paper",
                url=(
                    "https://example.com"
                ),
            )
        ]

        result = (
            ReportTools
            .format_report(
                title="AI Research",
                query=("Artificial Intelligence"),
                abstract=("AI abstract"),
                report_body=("AI analysis"),
                citations=citations,
            )
        )

        assert ("# AI Research" in result)
        assert ("## Abstract" in result)
        assert ("## References" in result)
        assert ("https://example.com" in result)

    def test_format_report_duplicate_citations(
        self,
    ):
        """Test duplicate citation removal."""

        citations = [
            Citation(
                doc_id="doc_1",
                claim="Claim 1",
                source="Source 1",
                url=(
                    "https://example.com"
                ),
            ),
            Citation(
                doc_id="doc_2",
                claim="Claim 2",
                source="Source 2",
                url=(
                    "https://example.com"
                ),
            ),
        ]

        result = (
            ReportTools
            .format_report(
                title="AI Research",
                query="AI",
                abstract="Abstract",
                report_body="Analysis",
                citations=citations,
            )
        )

        assert (
            result.count(
                "https://example.com"
            )
            == 1
        )

    def test_format_report_skips_empty_urls(
        self,
    ):
        """Test citations without URLs are skipped."""

        citations = [
            Citation(
                doc_id="doc_1",
                claim="Claim",
                source="Source",
                url="",
            )
        ]

        result = (
            ReportTools
            .format_report(
                title="AI Research",
                query="AI",
                abstract="Abstract",
                report_body="Analysis",
                citations=citations,
            )
        )

        assert (
            "URL:"
            not in result
        )