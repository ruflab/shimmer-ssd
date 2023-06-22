from simple_shapes_dataset.cli.graph import Graph, build_dependency_graph


def test_create_dependency_graph():
    domain_groups = [
        frozenset("v"),
        frozenset("t"),
        frozenset(["t", "v"]),
        frozenset(["a", "v"]),
        frozenset(["t", "v", "a"]),
    ]
    graph = build_dependency_graph(domain_groups)

    assert len(graph.nodes) == 5
    assert len(graph.edges) == 7


def test_graph_roots():
    graph = Graph(nodes={1, 2, 3, 4}, edges={(1, 2), (3, 2), (3, 4)})
    assert graph.get_roots() == {1, 3}
