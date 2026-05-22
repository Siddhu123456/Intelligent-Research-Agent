import pytest
from unittest.mock import Mock, patch

from tools.compression_tools import CompressionTools
from state.models import Document


class TestCompressionTools:
    """Unit tests for CompressionTools."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents."""

        return [
            Document(
                source="arxiv",
                title="AI Research",
                content=(
                    "Artificial Intelligence is "
                    "widely used in healthcare, "
                    "automation, and education."
                ),
                url="https://example.com/ai",
            ),
            Document(
                source="web",
                title="ML Trends",
                content=(
                    "Machine Learning adoption "
                    "continues to grow rapidly "
                    "across industries."
                ),
                url="https://example.com/ml",
            ),
        ]

    @patch(
        "tools.compression_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.compression_tools.JSONParser.safe_extract"
    )
    def test_compress_documents_success(
        self,
        mock_safe_extract,
        mock_create_llm,
        sample_documents,
    ):
        """Test successful document compression."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"content": '
            '"Relevant compressed content for testing purposes."}'
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
            "content":
            (
                "Relevant compressed "
                "content for testing purposes."
            )
        }

        with patch(
            "tools.compression_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            compressed_documents = (
                CompressionTools.compress_documents(
                    query="AI applications",
                    documents=sample_documents,
                )
            )

        assert (
            len(compressed_documents)
            == 2
        )

        assert all(
            isinstance(
                document,
                Document,
            )
            for document in compressed_documents
        )

    def test_compress_documents_empty_input(
        self,
    ):
        """Test compression with empty documents."""

        compressed_documents = (
            CompressionTools.compress_documents(
                query="AI",
                documents=[],
            )
        )

        assert compressed_documents == []

    @patch(
        "tools.compression_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.compression_tools.JSONParser.safe_extract"
    )
    def test_compress_documents_filters_short_content(
        self,
        mock_safe_extract,
        mock_create_llm,
        sample_documents,
    ):
        """Test filtering of short compressed content."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"content": "short"}'
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
            "content": "short"
        }

        with patch(
            "tools.compression_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            compressed_documents = (
                CompressionTools.compress_documents(
                    query="AI applications",
                    documents=sample_documents,
                )
            )

        assert compressed_documents == []

    @patch(
        "tools.compression_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.compression_tools.JSONParser.safe_extract"
    )
    def test_compress_documents_preserves_metadata(
        self,
        mock_safe_extract,
        mock_create_llm,
        sample_documents,
    ):
        """Test metadata preservation after compression."""

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"content": '
            '"Compressed relevant content for testing."}'
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
            "content":
            (
                "Compressed relevant "
                "content for testing."
            )
        }

        with patch(
            "tools.compression_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            compressed_documents = (
                CompressionTools.compress_documents(
                    query="AI applications",
                    documents=sample_documents,
                )
            )

        for (
            original,
            compressed,
        ) in zip(
            sample_documents,
            compressed_documents,
        ):
            assert (
                compressed.source
                == original.source
            )

            assert (
                compressed.title
                == original.title
            )

            assert (
                compressed.url
                == original.url
            )

    @patch(
        "tools.compression_tools.LLMFactory.create_qwen_llm"
    )
    @patch(
        "tools.compression_tools.JSONParser.safe_extract"
    )
    def test_compress_documents_respects_max_length(
        self,
        mock_safe_extract,
        mock_create_llm,
    ):
        """Test compressed content max length."""

        long_content = (
            "A"
            * (
                CompressionTools
                .MAX_DOCUMENT_LENGTH
                + 500
            )
        )

        documents = [
            Document(
                source="arxiv",
                title="Long Document",
                content=long_content,
                url="https://example.com",
            )
        ]

        mock_llm = Mock()

        mock_response = Mock()
        mock_response.content = (
            '{"content": "'
            + long_content +
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
            "content": long_content
        }

        with patch(
            "tools.compression_tools.ChatPromptTemplate.from_messages",
            return_value=mock_prompt,
        ):
            compressed_documents = (
                CompressionTools.compress_documents(
                    query="AI",
                    documents=documents,
                )
            )

        assert (
            len(
                compressed_documents[0]
                .content
            )
            <= CompressionTools
            .MAX_DOCUMENT_LENGTH
        )