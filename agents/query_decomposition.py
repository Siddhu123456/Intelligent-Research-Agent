from langsmith import traceable

from state.constants import CurrentStep, Domain
from state.schema import ResearchState
from tools.decompose_tools import (
    DecompositionTools,
)
from utils.logger import setup_logger


logger = setup_logger(__name__)


class QueryDecompositionAgent:
    """Agent responsible for research query decomposition."""

    @staticmethod
    @traceable(
        name="query_decomposition_agent",
    )
    def run(
        state: ResearchState,
    ) -> ResearchState:
        try:
            logger.info(
                "Starting query decomposition",
            )

            sub_queries = (
                DecompositionTools.decompose_query(
                    query=state["query"],
                )
            )
            
            # Limit arxiv usage

            arxiv_count = 0

            filtered_queries = []

            for sub_query in sub_queries:

                if (
                    sub_query.domain
                    == Domain.ARXIV
                ):

                    arxiv_count += 1
 
                    if arxiv_count > 3:

                        continue

                filtered_queries.append(
                    sub_query
                )

            sub_queries = filtered_queries

            is_valid = (
                DecompositionTools.validate_sub_queries(
                    sub_queries=sub_queries,
                )
            )

            if not is_valid:
                raise ValueError(
                    "Invalid sub-queries generated",
                )

            state["sub_queries"] = sub_queries

            state["current_step"] = (
                CurrentStep.DECOMPOSED.value
            )

            logger.info(
                "Query decomposition completed",
            )

            return state

        except Exception as error:
            logger.error(
                "Query decomposition failed: %s",
                str(error),
            )

            state["error"] = str(error)

            state["current_step"] = (
                CurrentStep.ERROR.value
            )

            return state