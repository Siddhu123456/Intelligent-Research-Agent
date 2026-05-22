from langchain_core.prompts import (
    ChatPromptTemplate,
)

from state.constants import (
    Domain,
)

from state.models import (
    SubQuery,
    SubQueryList,
)

from utils.llm_factory import (
    LLMFactory,
)


class DecompositionTools:
    """Tools for query decomposition."""

    MIN_SUB_QUERIES = 2

    MAX_SUB_QUERIES = 5

    @staticmethod
    def decompose_query(
        query: str,
    ) -> list[SubQuery]:
        """Decompose research query into retrieval sub-queries."""

        llm = (
            LLMFactory
            .create_qwen_llm(
                temperature=0.2,
            )
        )

        structured_llm = (
            llm.with_structured_output(
                SubQueryList,
            )
        )

        prompt = (
                    ChatPromptTemplate
                    .from_messages(
                        [
                            (
                                "system",
                                (
                                    "You are an advanced "
                                    "research query "
                                    "decomposition agent.\n\n"

                                    "Your task:\n"
                                    "Break the research query "
                                    "into high-quality retrieval "
                                    "sub-queries optimized for "
                                    "multi-source research.\n\n"

                                    "PRIMARY GOAL:\n"
                                    "Generate diverse retrieval "
                                    "coverage across multiple "
                                    "knowledge sources.\n\n"

                                    "RULES:\n"
                                    "- generate between 2 and 5 "
                                    "sub-queries\n"
                                    "- each query must target "
                                    "a different research aspect\n"
                                    "- keep each query concise "
                                    "and searchable\n"
                                    "- prefer keyword-rich "
                                    "search phrases\n"
                                    "- avoid repeating the "
                                    "same wording\n"
                                    "- avoid generic queries\n"
                                    "- avoid conversational "
                                    "phrasing\n"
                                    "- do not answer the query\n\n"

                                    "DOMAIN BALANCING:\n"
                                    "- ALWAYS include at least one WEB query\n"
                                    "- web should be the primary retrieval source\n"
                                    "- prefer WEB for practical "
                                    "and modern information\n"
                                    "- use wikipedia for "
                                    "foundational understanding\n"
                                    "- avoid domain bias\n"
                                    "- select domains based on "
                                    "query intent\n"
                                    "- distribute queries across "
                                    "multiple domains whenever "
                                    "possible\n\n"

                                    "AVAILABLE DOMAINS:\n\n"

                                    "1. web\n"
                                    "- tutorials\n"
                                    "- industry articles\n"
                                    "- practical applications\n"
                                    "- news\n"
                                    "- implementation guides\n"
                                    "- modern trends\n"
                                    "- real-world examples\n"
                                    "- case studies\n"
                                    "- tools and frameworks\n\n"

                                    "2. wikipedia\n"
                                    "- concepts\n"
                                    "- definitions\n"
                                    "- historical background\n"
                                    "- foundational understanding\n"
                                    "- terminology\n"
                                    "- theoretical explanations\n\n"

                                    "IMPORTANT:\n"
                                    "- most research queries "
                                    "should primarily use "
                                    "web retrieval\n"
                                    "- wikipedia should support "
                                    "conceptual understanding\n"
                                    "- prefer balanced retrieval "
                                    "coverage\n"
                                    "- avoid generating overly "
                                    "broad retrieval queries\n\n"

                                    "GOOD EXAMPLE:\n\n"

                                    "Query:\n"
                                    "'Quantum computing applications'\n\n"

                                    "Balanced Output:\n"
                                    "- quantum computing definition "
                                    "(wikipedia)\n"
                                    "- quantum computing use cases "
                                    "(web)\n"
                                    "- industry quantum adoption "
                                    "(web)\n"
                                    "- quantum computing history "
                                    "(wikipedia)\n\n"

                                    "Return structured output only."
                                ),
                            ),
                            (
                                "human",
                                "{query}",
                            ),
                        ]
                    )
                )

        chain = (
            prompt
            | structured_llm
        )

        response = chain.invoke(
            {
                "query": query,
            }
        )

        return response.sub_queries

    @staticmethod
    def validate_sub_queries(
        sub_queries: list[SubQuery],
    ) -> bool:
        """Validate decomposed sub-queries."""

        if not sub_queries:

            return False

        if (
            len(sub_queries)
            > DecompositionTools
            .MAX_SUB_QUERIES
        ):

            return False

        valid_domains = {
            Domain.WEB,
            Domain.ARXIV,
            Domain.WIKIPEDIA,
        }

        domains_used = {
            sub_query.domain
            for sub_query in sub_queries
        }

        # Require retrieval diversity

        if len(domains_used) < 2:

            return False

        return all(
            isinstance(
                sub_query,
                SubQuery,
            )
            and (
                sub_query.domain
                in valid_domains
            )
            for sub_query in (
                sub_queries
            )
        )