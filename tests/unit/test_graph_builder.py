from graph.builder import GraphBuilder


def test_build_graph_compiles_to_graph_with_async_methods():
    graph = GraphBuilder.build_graph()

    assert graph is not None
    assert hasattr(graph, "astream")
    assert hasattr(graph, "ainvoke")
