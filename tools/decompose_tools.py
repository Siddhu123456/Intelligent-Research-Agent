from langchain_core.prompts import (
    ChatPromptTemplate,
)

from state.constants import (
    Domain,
)

from state.models import (
    SubQuery,
)

from utils.json_parser import (
    JSONParser,
)

from utils.llm_factory import (
    LLMFactory,
)

from tools.prompts.decompose_tools_prompts import (
    DECOMPOSE_SYSTEM_PROMPT,
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

        prompt = (
            ChatPromptTemplate
            .from_messages(
                [
                    (
                        "system",
                        DECOMPOSE_SYSTEM_PROMPT,
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
            | llm
        )

        response = chain.invoke({"query": query})

        # Support both plain LLM content (response.content)
        # and structured pydantic outputs returned directly
        if hasattr(response, "content"):
            raw = response.content
        else:
            try:
                raw = response.dict()
            except Exception:
                raw = response

        # If the chain returned a structured dict/model, use it directly.
        if isinstance(raw, dict):
            parsed_response = raw
        else:
            parsed_response = JSONParser.safe_extract(
                content=raw,
                fallback={"sub_queries": []},
            )

        sub_queries_data = (
            parsed_response.get(
                "sub_queries",
                [],
            )
        )

        sub_queries: list[SubQuery] = []

        for item in sub_queries_data:

            try:

                sub_queries.append(
                    SubQuery(
                        query=item["query"],
                        domain=item["domain"],
                        priority=item.get(
                            "priority",
                            3,
                        ),
                    )
                )

            except Exception:

                continue

        return sub_queries

    @staticmethod
    def validate_sub_queries(
        sub_queries: list[SubQuery],
    ) -> bool:
        """Validate decomposed sub-queries."""

        if not sub_queries:

            return False

        if (
            len(sub_queries)
            < DecompositionTools
            .MIN_SUB_QUERIES
        ):

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
            for sub_query in sub_queries
        )