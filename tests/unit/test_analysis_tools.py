import pytest
from unittest.mock import Mock, patch

from tools.analysis_tools import AnalysisTools
from state.models import Document, Citation


class TestAnalysisTools:
    """Unit tests for AnalysisTools."""

    @pytest.fixture
    def sample_documents(self):
        """Sample document objects."""
        return [
            Document(
                id="1",
                title="AI Research Paper",
                content="Artificial Intelligence improves automation and efficiency.",
                source="arxiv",
                url="https://example.com/ai-paper",
            ),
            Document(
                id="2",
                title="ML Trends",
                content="Machine Learning adoption is increasing rapidly.",
                source="web",
                url="https://example.com/ml-trends",
            ),
        ]

    @patch(
        "tools.analysis_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.analysis_tools.JSONParser.safe_extract"
    )
    def test_extract_key_findings_success(
        self,
        mock_safe_extract,
        mock_create_llm,
        sample_documents,
    ):
        """Test successful key finding extraction."""

        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = (
            '{"findings": ["Finding 1", "Finding 2"]}'
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
            "findings": [
                "Finding 1",
                "Finding 2",
            ]
        }

        with patch(
            "tools.analysis_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            findings = (
                AnalysisTools.extract_key_findings(
                    documents=sample_documents,
                    query="AI trends",
                )
            )

        assert findings == [
            "Finding 1",
            "Finding 2",
        ]

    def test_extract_key_findings_empty_documents(
        self,
    ):
        """Test extract_key_findings with empty documents."""

        findings = (
            AnalysisTools.extract_key_findings(
                documents=[],
                query="AI trends",
            )
        )

        assert findings == []

    @patch(
        "tools.analysis_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.analysis_tools.JSONParser.safe_extract"
    )
    def test_generate_analysis_summary_success(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test summary generation."""

        findings = [
            "AI improves productivity.",
            "ML adoption is increasing.",
        ]

        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = (
            '{"summary": "AI and ML are growing rapidly."}'
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
            "AI and ML are growing rapidly."
        }

        with patch(
            "tools.analysis_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            summary = (
                AnalysisTools.generate_analysis_summary(
                    findings=findings
                )
            )

        assert (
            summary
            == "AI and ML are growing rapidly."
        )

    def test_generate_analysis_summary_empty_findings(
        self,
    ):
        """Test summary generation with empty findings."""

        summary = (
            AnalysisTools.generate_analysis_summary(
                findings=[]
            )
        )

        assert (
            summary
            == "Insufficient findings available for analysis."
        )

    @patch(
        "tools.analysis_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.analysis_tools.JSONParser.safe_extract"
    )
    def test_identify_contradictions_success(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test contradiction identification."""

        findings = [
            "AI improves accuracy.",
            "AI systems may introduce bias.",
        ]

        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = (
            '{"analysis": "Minor limitations detected."}'
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
            "analysis":
            "Minor limitations detected."
        }

        with patch(
            "tools.analysis_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            analysis = (
                AnalysisTools.identify_contradictions(
                    findings=findings
                )
            )

        assert (
            analysis
            == "Minor limitations detected."
        )

    def test_identify_contradictions_empty_findings(
        self,
    ):
        """Test contradiction analysis with empty findings."""

        analysis = (
            AnalysisTools.identify_contradictions(
                findings=[]
            )
        )

        assert (
            analysis
            == "No findings available for contradiction analysis."
        )

    def test_score_confidence_empty_documents(
        self,
    ):
        """Test confidence score with no documents."""

        confidence = (
            AnalysisTools.score_confidence(
                documents=[]
            )
        )

        assert confidence == 0.0

    def test_score_confidence_valid_documents(
        self,
        sample_documents,
    ):
        """Test confidence score calculation."""

        confidence = (
            AnalysisTools.score_confidence(
                documents=sample_documents
            )
        )

        assert isinstance(
            confidence,
            float,
        )

        assert 0.0 <= confidence <= 1.0

    def test_build_citations(
        self,
        sample_documents,
    ):
        """Test citation generation."""

        findings = [
            "AI improves automation.",
            "ML adoption is increasing.",
        ]

        citations = (
            AnalysisTools.build_citations(
                findings=findings,
                documents=sample_documents,
            )
        )

        assert len(citations) == 2

        assert all(
            isinstance(
                citation,
                Citation,
            )
            for citation in citations
        )

        assert (
            citations[0].claim
            == "AI improves automation."
        )

        assert (
            citations[0].source
            == "AI Research Paper"
        )

    def test_build_citations_empty_findings(
        self,
        sample_documents,
    ):
        """Test citation generation with empty findings."""

        citations = (
            AnalysisTools.build_citations(
                findings=[],
                documents=sample_documents,
            )
        )

        assert citations == []