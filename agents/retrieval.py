import asyncio

from utils.langsmith_wrapper import traceable

from state.constants import (
    CurrentStep,
    Domain,
)

from state.models import (
    Document,
)

from state.schema import (
    ResearchState,
)

from tools.chunking_tools import (
    ChunkingTools,
)

from tools.compression_tools import (
    CompressionTools,
)

from tools.reranking_tools import (
    RerankingTools,
)

from tools.search_tools import (
    SearchTools,
)

from tools.vector_tools import (
    VectorTools,
)

from utils.logger import (
    setup_logger,
)


logger = setup_logger(
    __name__,
)


class RetrievalAgent:
    """Agent responsible for multi-source retrieval."""

    MAX_SEMANTIC_RESULTS = 5

    MAX_RERANK_RESULTS = 5

    @staticmethod
    async def _retrieve_from_source(
        sub_query,
    ) -> list[Document]:
        """Retrieve documents from source."""

        try:

            if (
                sub_query.domain
                == Domain.WEB
            ):

                return await (
                    asyncio.to_thread(
                        SearchTools.search_web,
                        sub_query,
                    )
                )

            if (
                sub_query.domain
                == Domain.ARXIV
            ):

                return await (
                    asyncio.to_thread(
                        SearchTools.search_arxiv,
                        sub_query,
                    )
                )

            if (
                sub_query.domain
                == Domain.WIKIPEDIA
            ):

                return await (
                    asyncio.to_thread(
                        SearchTools
                        .search_wikipedia,
                        sub_query,
                    )
                )

            return []

        except Exception as error:

            logger.warning(
                "Retrieval source failed: %s",
                str(error),
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

            sub_queries = state[
                "sub_queries"
            ]

            # Run retrieval concurrently

            retrieval_tasks = [
                RetrievalAgent
                ._retrieve_from_source(
                    sub_query,
                )
                for sub_query in (
                    sub_queries
                )
            ]

            retrieval_results = (
                await asyncio.gather(
                    *retrieval_tasks,
                    return_exceptions=True,
                )
            )

            all_documents = []

            for result in (
                retrieval_results
            ):

                if isinstance(
                    result,
                    Exception,
                ):

                    logger.warning(
                        "Concurrent retrieval "
                        "task failed: %s",
                        str(result),
                    )

                    continue

                all_documents.extend(
                    result,
                )

            # Deduplicate documents

            deduplicated_documents = (
                SearchTools
                .deduplicate_documents(
                    all_documents,
                )
            )

            if (
                not deduplicated_documents
            ):

                logger.warning(
                    "No documents retrieved",
                )

                state[
                    "retrieved_documents"
                ] = []

                state["current_step"] = (
                    CurrentStep
                    .RETRIEVED
                    .value
                )

                return state

            # Chunk documents

            chunked_documents = (
                ChunkingTools
                .chunk_documents(
                    deduplicated_documents,
                )
            )

            # Store vectors

            VectorTools.store_documents(
                documents=chunked_documents,
                session_id=(
                    state["session_id"]
                ),
            )

            # Semantic retrieval

            semantic_documents = (
                VectorTools
                .semantic_search(
                    query=(
                        state[
                            "contextualized_query"
                        ]
                    ),

                    session_id=(
                        state["session_id"]
                    ),

                    top_k=(
                        RetrievalAgent
                        .MAX_SEMANTIC_RESULTS
                    ),
                )
            )
            
            # Rerank retrieved docs

            reranked_documents = (
                RerankingTools
                .rerank_documents(
                    query=(
                        state[
                            "contextualized_query"
                        ]
                    ),
                    documents=(
                        semantic_documents
                    ),
                    top_k=(
                        RetrievalAgent
                        .MAX_RERANK_RESULTS
                    ),
                )
            )

            # Contextual compression

            compressed_documents = (
                CompressionTools
                .compress_documents(
                    query=(
                        state[
                            "contextualized_query"
                        ]
                    ),
                    documents=(
                        reranked_documents
                    ),
                )
            )

            state[
                "retrieved_documents"
            ] = (
                compressed_documents
            )

            state["current_step"] = (
                CurrentStep
                .RETRIEVED
                .value
            )

            logger.info(
                "Retrieved %d raw documents",
                len(all_documents),
            )

            logger.info(
                "Deduplicated to %d documents",
                len(
                    deduplicated_documents
                ),
            )

            logger.info(
                "Retrieved %d semantic documents",
                len(
                    semantic_documents
                ),
            )

            logger.info(
                "Compressed to %d documents",
                len(
                    compressed_documents
                ),
            )

            return state

        except Exception as error:

            logger.error(
                "Retrieval failed: %s",
                str(error),
            )

            state["error"] = str(
                error,
            )

            state["current_step"] = (
                CurrentStep.ERROR.value
            )

            return state