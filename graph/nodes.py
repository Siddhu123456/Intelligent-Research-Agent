import asyncio
from state.constants import CurrentStep
from state.schema import ResearchState

from agents.query_decomposition import (
    QueryDecompositionAgent,
)
from agents.retrieval import RetrievalAgent
from agents.analysis import AnalysisAgent
from agents.report_generation import (
    ReportGenerationAgent,
)
from agents.context_agent import (
    ContextAgent,
)
from agents.supervisor_agent import (
    SupervisorAgent,
)



class GraphNodes:
    """Placeholder graph nodes for workflow orchestration."""

    @staticmethod
    def supervisor_node(
        state: ResearchState,
    ) -> ResearchState:
        return SupervisorAgent.run(
            state,
        )
    
    @staticmethod
    def context_node(
        state: ResearchState,
    ) -> ResearchState:
        return ContextAgent.run(
            state,
        )

    @staticmethod
    def query_decomposition_node(
        state: ResearchState,
    ) -> ResearchState:
        return (
            QueryDecompositionAgent.run(
                state,
            )
        )

    @staticmethod
    async def retrieval_node(
        state: ResearchState,
    ) -> ResearchState:
        return await RetrievalAgent.run(
                state,
        )

    @staticmethod
    def analysis_node(
        state: ResearchState,
    ) -> ResearchState:
        return AnalysisAgent.run(
            state,
        )

    @staticmethod
    def report_generation_node(
        state: ResearchState,
    ) -> ResearchState:
        return (
            ReportGenerationAgent.run(
                state,
            )
        )

    @staticmethod
    def error_recovery_node(
        state: ResearchState,
    ) -> ResearchState:
        state["current_step"] = (
            CurrentStep.ERROR.value
        )

        return state