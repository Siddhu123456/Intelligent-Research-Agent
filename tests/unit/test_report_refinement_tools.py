import pytest
from unittest.mock import Mock, patch

from tools.report_refinement_tools import (
    ReportRefinementTools,
)


class TestReportRefinementTools:
    """Unit tests for ReportRefinementTools."""

    @patch(
        "tools.report_refinement_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_refinement_tools.JSONParser.safe_extract"
    )
    def test_classify_refinement_intent_success(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test successful refinement intent classification."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"intent": "modify_section", '
            '"target_section": '
            '"analysis_and_insights"}'
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
            "intent":
            "modify_section",
            "target_section":
            "analysis_and_insights",
        }

        with patch(
            "tools.report_refinement_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            result = (
                ReportRefinementTools
                .classify_refinement_intent(
                    refinement_query=(
                        "Improve analysis"
                    ),
                    existing_sections=[
                        "introduction",
                        "analysis_and_insights",
                        "conclusion",
                    ],
                )
            )

        assert (
            result["intent"]
            == "modify_section"
        )

        assert (
            result["target_section"]
            == "analysis_and_insights"
        )

    @patch(
        "tools.report_refinement_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_refinement_tools.JSONParser.safe_extract"
    )
    def test_classify_refinement_intent_invalid_section(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test invalid target section correction."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"intent": "modify_section", '
            '"target_section": '
            '"invalid_section"}'
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
            "intent":
            "modify_section",
            "target_section":
            "invalid_section",
        }

        with patch(
            "tools.report_refinement_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            result = (
                ReportRefinementTools
                .classify_refinement_intent(
                    refinement_query=(
                        "Improve analysis"
                    ),
                    existing_sections=[
                        "introduction",
                        "analysis_and_insights",
                        "conclusion",
                    ],
                )
            )

        assert (
            result["target_section"]
            == "analysis_and_insights"
        )

    @patch(
        "tools.report_refinement_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_refinement_tools.JSONParser.safe_extract"
    )
    def test_refine_section_success(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test successful section refinement."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"updated_section": '
            '"Improved analysis section."}'
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
            "updated_section":
            (
                "Improved analysis "
                "section."
            )
        }

        with patch(
            "tools.report_refinement_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            result = (
                ReportRefinementTools
                .refine_section(
                    section_name=(
                        "analysis"
                    ),
                    section_content=(
                        "Old analysis"
                    ),
                    refinement_query=(
                        "Improve clarity"
                    ),
                )
            )

        assert (
            result
            == (
                "Improved analysis "
                "section."
            )
        )

    @patch(
        "tools.report_refinement_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_refinement_tools.JSONParser.safe_extract"
    )
    def test_refine_section_non_string(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test non-string updated section."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"updated_section": 12345}'
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
            "updated_section": 12345
        }

        with patch(
            "tools.report_refinement_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            result = (
                ReportRefinementTools
                .refine_section(
                    section_name=(
                        "analysis"
                    ),
                    section_content=(
                        "Old analysis"
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

        assert result == "12345"

    @patch(
        "tools.report_refinement_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_refinement_tools.JSONParser.safe_extract"
    )
    def test_generate_new_section_success(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test successful new section generation."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"new_section": '
            '"This section explains future work."}'
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
            "new_section":
            (
                "This section explains "
                "future work."
            )
        }

        with patch(
            "tools.report_refinement_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            result = (
                ReportRefinementTools
                .generate_new_section(
                    section_name=(
                        "future_work"
                    ),
                    report_topic=(
                        "AI Research"
                    ),
                    refinement_query=(
                        "Add future work"
                    ),
                    existing_sections=[
                        "introduction",
                        "results",
                    ],
                )
            )

        assert (
            result
            == (
                "This section explains "
                "future work."
            )
        )

    @patch(
        "tools.report_refinement_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_refinement_tools.JSONParser.safe_extract"
    )
    def test_generate_new_section_empty_response(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test empty generated section fallback."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"new_section": ""}'
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
            "new_section": ""
        }

        with patch(
            "tools.report_refinement_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            result = (
                ReportRefinementTools
                .generate_new_section(
                    section_name=(
                        "future_work"
                    ),
                    report_topic=(
                        "AI Research"
                    ),
                    refinement_query=(
                        "Add future work"
                    ),
                    existing_sections=[
                        "introduction",
                        "results",
                    ],
                )
            )

        assert (
            result
            == (
                "Additional discussion on "
                "future work."
            )
        )

    @patch(
        "tools.report_refinement_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.report_refinement_tools.JSONParser.safe_extract"
    )
    def test_determine_section_placement_success(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test section placement determination."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"insert_before": '
            '"conclusion"}'
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
            "insert_before":
            "conclusion"
        }

        with patch(
            "tools.report_refinement_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            result = (
                ReportRefinementTools
                .determine_section_placement(
                    new_section=(
                        "Future work section"
                    ),
                    existing_sections=[
                        "introduction",
                        "results",
                        "conclusion",
                    ],
                )
            )

        assert (
            result["insert_before"]
            == "conclusion"
        )