import hashlib

import arxiv
import wikipediaapi
from tavily import TavilyClient

from config.settings import settings
from state.constants import Domain
from state.models import Document
from state.models import SubQuery
from utils.logger import setup_logger


logger = setup_logger(
    __name__,
)

class SearchTools:
    """Tools for multi-source retrieval."""

    _tavily_client = TavilyClient(
        api_key=settings.tavily_api_key,
    )

    _wikipedia_client = wikipediaapi.Wikipedia(
        user_agent="research-assistant",
        language="en",
    )

    @staticmethod
    def search_web(
        sub_query: SubQuery,
        max_results: int = 5,
    ) -> list[Document]:
        
        logger.info(
            "Starting Tavily search: %s",
            sub_query.query,
        )
        
        response = (
            SearchTools._tavily_client.search(
                query=sub_query.query,
                max_results=max_results,
            )
        )

        documents: list[Document] = []

        for result in response["results"]:
            documents.append(
                Document(
                    source=Domain.WEB,
                    title=result["title"],
                    content=result["content"][
                        :4000
                    ],
                    url=result["url"],
                )
            )

        return documents

    @staticmethod
    def search_arxiv(
        sub_query: SubQuery,
        max_results: int = 3,
    ) -> list[Document]:
        client = arxiv.Client(
            page_size=max_results,
            delay_seconds=3,
            num_retries=2,
        )

        search = arxiv.Search(
            query=sub_query.query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        documents: list[Document] = []

        try:
            for paper in client.results(search):
                documents.append(
                    Document(
                        source=Domain.ARXIV,
                        title=paper.title,
                        content=paper.summary[:4000],
                        url=paper.pdf_url,
                    )
                )

        except Exception:
            return []

        return documents

    @staticmethod
    def search_wikipedia(
        sub_query: SubQuery,
    ) -> list[Document]:
        page = (
            SearchTools._wikipedia_client.page(
                sub_query.query,
            )
        )

        if not page.exists():
            return []

        return [
            Document(
                source=Domain.WIKIPEDIA,
                title=page.title,
                content=page.summary[:4000],
                url=page.fullurl,
            )
        ]

    @staticmethod
    def deduplicate_documents(
        documents: list[Document],
    ) -> list[Document]:
        unique_documents: list[Document] = []

        seen_hashes: set[str] = set()

        for document in documents:
            if not document.url:
                continue

            document_hash = hashlib.sha256(
                document.url.encode(),
            ).hexdigest()

            if document_hash in seen_hashes:
                continue

            seen_hashes.add(document_hash)

            unique_documents.append(
                document,
            )

        return unique_documents