import asyncio

from langsmith import traceable

from state.constants import CurrentStep, Domain
from state.models import Document
from state.schema import ResearchState
from tools.search_tools import SearchTools
from tools.vector_tools import (
    VectorTools,
)
from tools.chunking_tools import (
    ChunkingTools,
)
from tools.reranking_tools import (
    RerankingTools,
)
from tools.compression_tools import (
    CompressionTools,
)
from utils.logger import setup_logger


logger = setup_logger(__name__)


class RetrievalAgent:
    """Agent responsible for multi-source retrieval."""

    @staticmethod
    async def _retrieve_from_source(
        sub_query,
    ) -> list[Document]:
        if sub_query.domain == Domain.WEB:
            return await asyncio.to_thread(
                SearchTools.search_web,
                sub_query,
            )

        if sub_query.domain == Domain.ARXIV:
            return await asyncio.to_thread(
                SearchTools.search_arxiv,
                sub_query,
            )

        if (
            sub_query.domain
            == Domain.WIKIPEDIA
        ):
            return await asyncio.to_thread(
                SearchTools.search_wikipedia,
                sub_query,
            )

        return []

    @staticmethod
    @traceable(
        name="retrieval_agent",
    )
    async def run(
        state: ResearchState,
    ) -> ResearchState:
        try:
            logger.info(
                "Starting retrieval process",
            )

            all_documents: list[Document] = []

            for sub_query in state["sub_queries"]:
                try:
                    documents = (
                        await RetrievalAgent
                        ._retrieve_from_source(
                            sub_query,
                        )
                    )

                    all_documents.extend(
                        documents,
                    )

                except Exception as error:
                    logger.warning(
                        (
                            "Failed retrieval for "
                            "query '%s': %s"
                        ),
                        sub_query.query,
                        str(error),
                    )

                    continue

            deduplicated_documents = (
                SearchTools.deduplicate_documents(
                    all_documents,
                )
            )
            
            chunked_documents = (
                ChunkingTools.chunk_documents(
                    deduplicated_documents,
                )
            )
            
            VectorTools.store_documents(
                chunked_documents,
            )
            
            semantic_documents = (
                VectorTools.semantic_search(
                    query=state["query"],
                    top_k=5,
                )
            )
            
            reranked_documents = (
                RerankingTools.rerank_documents(
                    query=(
                        state[
                            "contextualized_query"
                        ]
                    ),
                    documents=semantic_documents,
                    top_k=5,
                )
            )
            
            compressed_documents = (
                CompressionTools.compress_documents(
                    query=(
                        state[
                            "contextualized_query"
                        ]
                    ),
                    documents=reranked_documents,
                )
            )
            
            state["retrieved_documents"] = (
                compressed_documents
            )

            state["current_step"] = (
                CurrentStep.RETRIEVED.value
            )

            logger.info(
                "Stored %d documents in vector DB",
                len(deduplicated_documents),
            )

            logger.info(
                "Retrieved %d semantic documents",
                len(semantic_documents),
            )

            return state

        except Exception as error:
            logger.error(
                "Retrieval failed: %s",
                str(error),
            )

            state["error"] = str(error)

            state["current_step"] = (
                CurrentStep.ERROR.value
            )

            return state