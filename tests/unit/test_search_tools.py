import pytest
from unittest.mock import Mock, patch

from tools.search_tools import (
    SearchTools,
)
from state.constants import Domain
from state.models import (
    Document,
    SubQuery,
)


class TestSearchTools:
    """Unit tests for SearchTools."""

    @pytest.fixture
    def sample_sub_query(self):
        """Create sample sub query."""

        return SubQuery(
            query="Artificial Intelligence",
            domain=Domain.WEB,
            priority=1,
        )

    @patch(
        "tools.search_tools.SearchTools._tavily_client.search"
    )
    def test_search_web_success(
        self,
        mock_search,
        sample_sub_query,
    ):
        """Test successful Tavily search."""

        mock_search.return_value = {
            "results": [
                {
                    "title":
                    "AI Research",
                    "content":
                    (
                        "Artificial Intelligence "
                        "research content."
                    ),
                    "url":
                    "https://example.com/ai",
                },
                {
                    "title":
                    "ML Trends",
                    "content":
                    (
                        "Machine Learning "
                        "trends content."
                    ),
                    "url":
                    "https://example.com/ml",
                },
            ]
        }

        result = (
            SearchTools.search_web(
                sub_query=sample_sub_query,
                max_results=2,
            )
        )

        assert len(result) == 2

        assert all(
            isinstance(
                document,
                Document,
            )
            for document in result
        )

        assert (
            result[0].source
            == Domain.WEB
        )

    @patch(
        "tools.search_tools.arxiv.Client"
    )
    @patch(
        "tools.search_tools.arxiv.Search"
    )
    def test_search_arxiv_success(
        self,
        mock_search,
        mock_client,
        sample_sub_query,
    ):
        """Test successful arxiv search."""

        mock_paper = Mock()

        mock_paper.title = (
            "Quantum Computing"
        )

        mock_paper.summary = (
            "Quantum computing "
            "research summary."
        )

        mock_paper.pdf_url = (
            "https://arxiv.org/pdf/1234"
        )

        mock_client_instance = Mock()

        mock_client_instance.results.return_value = [
            mock_paper
        ]

        mock_client.return_value = (
            mock_client_instance
        )

        result = (
            SearchTools.search_arxiv(
                sub_query=sample_sub_query,
                max_results=1,
            )
        )

        assert len(result) == 1

        assert (
            result[0].source
            == Domain.ARXIV
        )

        assert (
            result[0].title
            == "Quantum Computing"
        )

    @patch(
        "tools.search_tools.arxiv.Client"
    )
    @patch(
        "tools.search_tools.arxiv.Search"
    )
    def test_search_arxiv_exception(
        self,
        mock_search,
        mock_client,
        sample_sub_query,
    ):
        """Test arxiv exception handling."""

        mock_client_instance = Mock()

        mock_client_instance.results.side_effect = (
            Exception(
                "Arxiv API failure"
            )
        )

        mock_client.return_value = (
            mock_client_instance
        )

        result = (
            SearchTools.search_arxiv(
                sub_query=sample_sub_query,
            )
        )

        assert result == []

    @patch(
        "tools.search_tools.SearchTools._wikipedia_client.page"
    )
    def test_search_wikipedia_success(
        self,
        mock_page,
        sample_sub_query,
    ):
        """Test successful Wikipedia search."""

        mock_wiki_page = Mock()

        mock_wiki_page.exists.return_value = (
            True
        )

        mock_wiki_page.title = (
            "Artificial Intelligence"
        )

        mock_wiki_page.summary = (
            "AI summary content."
        )

        mock_wiki_page.fullurl = (
            "https://wikipedia.org/ai"
        )

        mock_page.return_value = (
            mock_wiki_page
        )

        result = (
            SearchTools.search_wikipedia(
                sub_query=sample_sub_query,
            )
        )

        assert len(result) == 1

        assert (
            result[0].source
            == Domain.WIKIPEDIA
        )

    @patch(
        "tools.search_tools.SearchTools._wikipedia_client.page"
    )
    def test_search_wikipedia_page_not_found(
        self,
        mock_page,
        sample_sub_query,
    ):
        """Test Wikipedia page not found."""

        mock_wiki_page = Mock()

        mock_wiki_page.exists.return_value = (
            False
        )

        mock_page.return_value = (
            mock_wiki_page
        )

        result = (
            SearchTools.search_wikipedia(
                sub_query=sample_sub_query,
            )
        )

        assert result == []

    def test_deduplicate_documents_success(
        self,
    ):
        """Test document deduplication."""

        documents = [
            Document(
                source=Domain.WEB,
                title="Doc 1",
                content="Content 1",
                url="https://example.com",
            ),
            Document(
                source=Domain.WEB,
                title="Doc 2",
                content="Content 2",
                url="https://example.com",
            ),
            Document(
                source=Domain.ARXIV,
                title="Doc 3",
                content="Content 3",
                url="https://arxiv.org/1234",
            ),
        ]

        result = (
            SearchTools
            .deduplicate_documents(
                documents
            )
        )

        assert len(result) == 2

    def test_deduplicate_documents_empty(
        self,
    ):
        """Test empty document deduplication."""

        result = (
            SearchTools
            .deduplicate_documents(
                []
            )
        )

        assert result == []

    def test_deduplicate_documents_skips_empty_url(
        self,
    ):
        """Test documents without URLs are skipped."""

        documents = [
            Document(
                source=Domain.WEB,
                title="Doc 1",
                content="Content 1",
                url="",
            ),
            Document(
                source=Domain.ARXIV,
                title="Doc 2",
                content="Content 2",
                url="https://arxiv.org/1234",
            ),
        ]

        result = (
            SearchTools
            .deduplicate_documents(
                documents
            )
        )

        assert len(result) == 1

        assert (
            result[0].url
            == "https://arxiv.org/1234"
        )