from langgraph.graph import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph.nodes import GraphNodes
from graph.router import GraphRouter
from state.schema import ResearchState


class GraphBuilder:
    """Builds and compiles the workflow graph."""

    @staticmethod
    def build_graph() -> CompiledStateGraph:
        graph = StateGraph(ResearchState)

        graph.add_node(
            "supervisor",
            GraphNodes.supervisor_node,
        )

        graph.add_node(
            "context",
            GraphNodes.context_node,
        )
        
        graph.add_node(
            "query_decomposition",
            GraphNodes.query_decomposition_node,
        )

        graph.add_node(
            "retrieval",
            GraphNodes.retrieval_node,
        )

        graph.add_node(
            "analysis",
            GraphNodes.analysis_node,
        )

        graph.add_node(
            "report_generation",
            GraphNodes.report_generation_node,
        )

        graph.add_node(
            "error_recovery",
            GraphNodes.error_recovery_node,
        )

        graph.set_entry_point(
            "supervisor",
        )

        graph.add_conditional_edges(
            "supervisor",
            GraphRouter.route,
        )
        
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
            END,
        )

        graph.add_edge(
            "error_recovery",
            END,
        )

        return graph.compile()