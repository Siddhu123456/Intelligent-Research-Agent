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
                        "You are an expert research "
                        "query decomposition agent.\n"
                        "Break the user query into "
                        "2 to 5 focused research "
                        "sub-queries.\n"
                        "Assign an appropriate domain "
                        "to each query.\n"
                        "Domains available:\n"
                        "- web\n"
                        "- arxiv\n"
                        "- wikipedia\n\n"
                        "Generate concise retrieval-optimized "
                        "queries.\n"
                        "Maximum 6 words per sub-query.\n"
                        "Avoid long sentence-style queries.\n"
                        "Return ONLY valid structured "
                        "output."
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

        return all(
            isinstance(
                sub_query,
                SubQuery,
            )
            for sub_query in sub_queries
        )