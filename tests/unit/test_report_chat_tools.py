import pytest
from unittest.mock import Mock, patch

from tools.report_chat_tools import (
    ReportChatTools,
)


class TestReportChatTools:
    """Unit tests for ReportChatTools."""

    @patch(
        "tools.report_chat_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_chat_tools.JSONParser.safe_extract"
    )
    def test_answer_report_question_success(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test successful question answering."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"answer": '
            '"AI improves automation efficiency."}'
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
            "answer":
            (
                "AI improves automation "
                "efficiency."
            )
        }

        with patch(
            "tools.report_chat_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            answer = (
                ReportChatTools
                .answer_report_question(
                    report=(
                        "AI is used in "
                        "automation systems."
                    ),
                    question=(
                        "What does AI improve?"
                    ),
                )
            )

        assert (
            answer
            == (
                "AI improves automation "
                "efficiency."
            )
        )

    def test_answer_report_question_empty_report(
        self,
    ):
        """Test empty report handling."""

        answer = (
            ReportChatTools
            .answer_report_question(
                report="",
                question=(
                    "What is AI?"
                ),
            )
        )

        assert (
            answer
            == (
                "No active report is "
                "available for question "
                "answering."
            )
        )

    @patch(
        "tools.report_chat_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_chat_tools.JSONParser.safe_extract"
    )
    def test_answer_report_question_non_string_answer(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test non-string answer conversion."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"answer": 12345}'
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
            "answer": 12345
        }

        with patch(
            "tools.report_chat_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            answer = (
                ReportChatTools
                .answer_report_question(
                    report=(
                        "AI systems data"
                    ),
                    question=(
                        "Give number"
                    ),
                )
            )

        assert isinstance(
            answer,
            str,
        )

        assert answer == "12345"

    @patch(
        "tools.report_chat_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_chat_tools.JSONParser.safe_extract"
    )
    def test_answer_report_question_max_response_length(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test response length truncation."""

        long_answer = (
            "A"
            * (
                ReportChatTools
                .MAX_RESPONSE_LENGTH
                + 500
            )
        )

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"answer": "' +
            long_answer +
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
            "answer": long_answer
        }

        with patch(
            "tools.report_chat_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            answer = (
                ReportChatTools
                .answer_report_question(
                    report=(
                        "AI systems report"
                    ),
                    question=(
                        "Explain AI"
                    ),
                )
            )

        assert (
            len(answer)
            <= ReportChatTools
            .MAX_RESPONSE_LENGTH
        )

    @patch(
        "tools.report_chat_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_chat_tools.JSONParser.safe_extract"
    )
    def test_answer_report_question_fallback_response(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test fallback response handling."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            "invalid json"
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
            "answer":
            (
                "Unable to generate "
                "a grounded answer "
                "from the workspace "
                "memory."
            )
        }

        with patch(
            "tools.report_chat_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            answer = (
                ReportChatTools
                .answer_report_question(
                    report=(
                        "Research data"
                    ),
                    question=(
                        "Explain findings"
                    ),
                )
            )

        assert (
            answer
            == (
                "Unable to generate "
                "a grounded answer "
                "from the workspace "
                "memory."
            )
        )