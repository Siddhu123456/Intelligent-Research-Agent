from state.constants import (
    CurrentStep,
)

from state.schema import (
    ResearchState,
)

from agents.analysis import (
    AnalysisAgent,
)

from agents.context_agent import (
    ContextAgent,
)

from agents.document_generation_agent import (
    DocumentGenerationAgent,
)

from agents.query_decomposition import (
    QueryDecompositionAgent,
)

from agents.report_chat_agent import (
    ReportChatAgent,
)

from agents.report_generation import (
    ReportGenerationAgent,
)

from agents.report_refinement_agent import (
    ReportRefinementAgent,
)

from agents.retrieval import (
    RetrievalAgent,
)

from agents.supervisor_agent import (
    SupervisorAgent,
)


class GraphNodes:
    """Workflow graph nodes."""

    @staticmethod
    def supervisor_node(
        state: ResearchState,
    ) -> ResearchState:

        return (
            SupervisorAgent.run(
                state,
            )
        )

    @staticmethod
    def context_node(
        state: ResearchState,
    ) -> ResearchState:

        return (
            ContextAgent.run(
                state,
            )
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

        return await (
            RetrievalAgent.run(
                state,
            )
        )

    @staticmethod
    def analysis_node(
        state: ResearchState,
    ) -> ResearchState:

        return (
            AnalysisAgent.run(
                state,
            )
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
    def report_refinement_node(
        state: ResearchState,
    ) -> ResearchState:

        return (
            ReportRefinementAgent.run(
                state,
            )
        )

    @staticmethod
    def report_chat_node(
        state: ResearchState,
    ) -> ResearchState:

        return (
            ReportChatAgent.run(
                state,
            )
        )

    @staticmethod
    def document_generation_node(
        state: ResearchState,
    ) -> ResearchState:

        return (
            DocumentGenerationAgent.run(
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