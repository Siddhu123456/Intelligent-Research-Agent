import pytest
from unittest.mock import Mock, patch

from tools.report_compression_tools import (
    ReportCompressionTools,
)


class TestReportCompressionTools:
    """Unit tests for ReportCompressionTools."""

    @patch(
        "tools.report_compression_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_compression_tools.JSONParser.safe_extract"
    )
    def test_compress_report_success(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test successful report compression."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"compressed_context": '
            '"Compressed AI research summary."}'
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
            "compressed_context":
            (
                "Compressed AI "
                "research summary."
            )
        }

        with patch(
            "tools.report_compression_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            result = (
                ReportCompressionTools
                .compress_report(
                    report=(
                        "Artificial Intelligence "
                        "research content."
                    )
                )
            )

        assert (
            result
            == (
                "Compressed AI "
                "research summary."
            )
        )

    def test_compress_report_empty(
        self,
    ):
        """Test empty report compression."""

        result = (
            ReportCompressionTools
            .compress_report(
                report=""
            )
        )

        assert (
            result
            == (
                "No report available "
                "for compression."
            )
        )

    @patch(
        "tools.report_compression_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_compression_tools.JSONParser.safe_extract"
    )
    def test_compress_report_non_string(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test non-string compressed context."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"compressed_context": 12345}'
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
            "compressed_context": 12345
        }

        with patch(
            "tools.report_compression_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            result = (
                ReportCompressionTools
                .compress_report(
                    report=(
                        "AI report"
                    )
                )
            )

        assert isinstance(
            result,
            str,
        )

        assert result == "12345"

    @patch(
        "tools.report_compression_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_compression_tools.JSONParser.safe_extract"
    )
    def test_compress_report_max_length(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test max compressed length."""

        long_text = (
            "A"
            * (
                ReportCompressionTools
                .MAX_COMPRESSED_LENGTH
                + 500
            )
        )

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"compressed_context": "'
            + long_text +
            '"}'
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
            "compressed_context":
            long_text
        }

        with patch(
            "tools.report_compression_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            result = (
                ReportCompressionTools
                .compress_report(
                    report=(
                        "AI report"
                    )
                )
            )

        assert (
            len(result)
            <= ReportCompressionTools
            .MAX_COMPRESSED_LENGTH
        )

    @patch(
        "tools.report_compression_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_compression_tools.JSONParser.safe_extract"
    )
    def test_update_compressed_context_success(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test successful context update."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"updated_context": '
            '"Updated AI workspace memory."}'
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
            "updated_context":
            (
                "Updated AI "
                "workspace memory."
            )
        }

        with patch(
            "tools.report_compression_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            result = (
                ReportCompressionTools
                .update_compressed_context(
                    previous_context=(
                        "Previous memory"
                    ),
                    updated_section_name=(
                        "Results"
                    ),
                    updated_section_content=(
                        "New findings"
                    ),
                    refinement_query=(
                        "Add findings"
                    ),
                )
            )

        assert (
            result
            == (
                "Updated AI "
                "workspace memory."
            )
        )

    @patch(
        "tools.report_compression_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_compression_tools.JSONParser.safe_extract"
    )
    def test_update_compressed_context_non_string(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test non-string updated context."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"updated_context": 999}'
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
            "updated_context": 999
        }

        with patch(
            "tools.report_compression_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            result = (
                ReportCompressionTools
                .update_compressed_context(
                    previous_context=(
                        "Previous memory"
                    ),
                    updated_section_name=(
                        "Discussion"
                    ),
                    updated_section_content=(
                        "Updated discussion"
                    ),
                    refinement_query=(
                        "Improve clarity"
                    ),
                )
            )

        assert isinstance(
            result,
            str,
        )

        assert result == "999"

    @patch(
        "tools.report_compression_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_compression_tools.JSONParser.safe_extract"
    )
    def test_update_compressed_context_max_length(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test updated context max length."""

        long_text = (
            "B" * 5000
        )

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"updated_context": "'
            + long_text +
            '"}'
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
            "updated_context":
            long_text
        }

        with patch(
            "tools.report_compression_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            result = (
                ReportCompressionTools
                .update_compressed_context(
                    previous_context=(
                        "Previous memory"
                    ),
                    updated_section_name=(
                        "Conclusion"
                    ),
                    updated_section_content=(
                        "Updated conclusion"
                    ),
                    refinement_query=(
                        "Summarize"
                    ),
                )
            )

        assert (
            len(result)
            <= 4000
        )