from langchain_core.prompts import ChatPromptTemplate

from state.constants import Domain
from state.models import SubQuery
from state.models import SubQueryList
from utils.llm_factory import LLMFactory


class DecompositionTools:
    """Tools for query decomposition."""

    @staticmethod
    def decompose_query(
        query: str,
    ) -> list[SubQuery]:

        llm = (
            LLMFactory.create_qwen_llm(
                temperature=0.2,
            )
        )

        structured_llm = (
            llm.with_structured_output(
                SubQueryList,
            )
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a research query decomposition agent.\n\n"

                        "Break the user query into 2 to 5 short "
                        "research sub-queries.\n\n"

                        "RULES:\n"
                        "- Use at least 2 different domains.\n"
                        "- Do not use the same domain for all queries.\n"
                        "- Keep each query under 6 words.\n"
                        "- Prefer concise searchable phrases.\n\n"

                        "AVAILABLE DOMAINS:\n"
                        "- arxiv\n"
                        "- web\n"
                        "- wikipedia\n\n"

                        "Use:\n"
                        "- arxiv for research topics\n"
                        "- web for tutorials/articles\n"
                        "- wikipedia for concepts/definitions\n\n"

                        "Return valid structured output only."
                    ),
                ),
                (
                    "human",
                    "{query}",
                ),
            ]
        )

        chain = prompt | structured_llm

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

        if not sub_queries:
            return False

        if len(sub_queries) > 5:
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

        # Ensure at least 2 different domains
        if len(domains_used) < 2:
            return False

        return all(
            isinstance(
                sub_query,
                SubQuery,
            )
            and sub_query.domain in valid_domains
            for sub_query in sub_queries
        )