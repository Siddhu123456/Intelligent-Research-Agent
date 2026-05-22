import pytest

from tools.chunking_tools import ChunkingTools
from state.models import Document


class TestChunkingTools:
    """Unit tests for ChunkingTools."""

    @pytest.fixture
    def sample_document(self):
        """Create sample document."""

        return Document(
            source="arxiv",
            title="AI Research",
            content="A" * 2500,
            url="https://example.com/ai",
        )

    def test_chunk_documents_success(
        self,
        sample_document,
    ):
        """Test successful document chunking."""

        chunked_documents = (
            ChunkingTools.chunk_documents(
                documents=[sample_document]
            )
        )

        assert (
            len(chunked_documents) > 1
        )

        assert all(
            isinstance(
                document,
                Document,
            )
            for document in chunked_documents
        )

    def test_chunk_documents_empty_input(
        self,
    ):
        """Test chunking with empty document list."""

        chunked_documents = (
            ChunkingTools.chunk_documents(
                documents=[]
            )
        )

        assert chunked_documents == []

    def test_chunk_documents_preserves_metadata(
        self,
        sample_document,
    ):
        """Test metadata preservation after chunking."""

        chunked_documents = (
            ChunkingTools.chunk_documents(
                documents=[sample_document]
            )
        )

        for chunk in chunked_documents:
            assert (
                chunk.source
                == sample_document.source
            )

            assert (
                chunk.url
                == sample_document.url
            )

            assert (
                sample_document.title
                in chunk.title
            )

    def test_chunk_documents_title_format(
        self,
        sample_document,
    ):
        """Test chunk title formatting."""

        chunked_documents = (
            ChunkingTools.chunk_documents(
                documents=[sample_document]
            )
        )

        for index, chunk in enumerate(
            chunked_documents,
        ):
            expected_title = (
                f"{sample_document.title} "
                f"(Chunk {index + 1})"
            )

            assert (
                chunk.title
                == expected_title
            )

    def test_chunk_documents_chunk_content_not_empty(
        self,
        sample_document,
    ):
        """Test chunk contents are not empty."""

        chunked_documents = (
            ChunkingTools.chunk_documents(
                documents=[sample_document]
            )
        )

        assert all(
            chunk.content.strip()
            for chunk in chunked_documents
        )

    def test_chunk_documents_single_small_document(
        self,
    ):
        """Test chunking small document."""

        document = Document(
            source="web",
            title="Small Doc",
            content="Short content",
            url="https://example.com",
        )

        chunked_documents = (
            ChunkingTools.chunk_documents(
                documents=[document]
            )
        )

        assert (
            len(chunked_documents) == 1
        )

        assert (
            chunked_documents[0].content
            == "Short content"
        )

    def test_chunk_documents_multiple_documents(
        self,
    ):
        """Test chunking multiple documents."""

        documents = [
            Document(
                source="arxiv",
                title="Doc 1",
                content="A" * 1500,
                url="https://example1.com",
            ),
            Document(
                source="web",
                title="Doc 2",
                content="B" * 1500,
                url="https://example2.com",
            ),
        ]

        chunked_documents = (
            ChunkingTools.chunk_documents(
                documents=documents
            )
        )

        assert (
            len(chunked_documents) >= 2
        )

        titles = [
            chunk.title
            for chunk in chunked_documents
        ]

        assert any(
            "Doc 1" in title
            for title in titles
        )

        assert any(
            "Doc 2" in title
            for title in titles
        )