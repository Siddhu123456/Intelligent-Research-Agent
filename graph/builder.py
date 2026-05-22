from langgraph.graph import (
    END,
    StateGraph,
)

from langgraph.graph.state import (
    CompiledStateGraph,
)

from graph.nodes import (
    GraphNodes,
)

from graph.router import (
    GraphRouter,
)

from state.schema import (
    ResearchState,
)


class GraphBuilder:
    """Build and compile workflow graph."""

    @staticmethod
    def build_graph() -> (
        CompiledStateGraph
    ):

        graph = StateGraph(
            ResearchState,
        )

        # Supervisor node

        graph.add_node(
            "supervisor",
            GraphNodes.supervisor_node,
        )

        # Core workflow nodes

        graph.add_node(
            "context",
            GraphNodes.context_node,
        )

        graph.add_node(
            "query_decomposition",
            (
                GraphNodes
                .query_decomposition_node
            ),
        )

        graph.add_node(
            "retrieval",
            GraphNodes.retrieval_node,
        )

        graph.add_node(
            "analysis",
            GraphNodes.analysis_node,
        )

        # Workspace nodes

        graph.add_node(
            "report_generation",
            (
                GraphNodes
                .report_generation_node
            ),
        )

        graph.add_node(
            "report_refinement",
            (
                GraphNodes
                .report_refinement_node
            ),
        )

        graph.add_node(
            "report_chat",
            GraphNodes.report_chat_node,
        )

        graph.add_node(
            "document_generation",
            (
                GraphNodes
                .document_generation_node
            ),
        )

        # Error handling node

        graph.add_node(
            "error_recovery",
            (
                GraphNodes
                .error_recovery_node
            ),
        )

        # Entry point

        graph.set_entry_point(
            "supervisor",
        )

        # Conditional routing

        graph.add_conditional_edges(
            "supervisor",
            GraphRouter.route,
        )

        # Workflow edges

        graph.add_edge(
            "context",
            "supervisor",
        )

        graph.add_edge(
            "query_decomposition",
            "supervisor",
        )

        graph.add_edge(
            "retrieval",
            "supervisor",
        )

        graph.add_edge(
            "analysis",
            "supervisor",
        )

        graph.add_edge(
            "report_generation",
            "supervisor",
        )

        graph.add_edge(
            "report_refinement",
            "supervisor",
        )

        graph.add_edge(
            "report_chat",
            "supervisor",
        )

        graph.add_edge(
            "document_generation",
            "supervisor",
        )

        # Error termination

        graph.add_edge(
            "error_recovery",
            END,
        )

        return graph.compile()