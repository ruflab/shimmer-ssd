from typing import Generic, TypeVar

T = TypeVar("T")


class Graph(Generic[T]):
    def __init__(
        self,
        nodes: set[T] | None = None,
        edges: set[tuple[T, T]] | None = None,
    ):
        self.nodes: set[T] = nodes or set()
        self.edges: set[tuple[T, T]] = edges or set()

    def add_node(self, node: T) -> None:
        self.nodes.add(node)

    def add_edge(self, edge: tuple[T, T]) -> None:
        self.edges.add(edge)

    def remove_nodes(self, nodes: set[T]) -> None:
        for edge in self.edges.copy():
            if edge[0] in nodes or edge[1] in nodes:
                self.edges.remove(edge)
        self.nodes -= nodes

    def get_roots(self) -> set[T]:
        roots = set()
        for node in self.nodes:
            if not self.has_edge_to(node):
                roots.add(node)
        return roots

    def has_edge_from(self, node: T) -> bool:
        return any(source_node == node for source_node, _ in self.edges)

    def has_edge_to(self, node: T) -> bool:
        return any(target_node == node for _, target_node in self.edges)

    def children(self, node: T) -> set[T]:
        children = set()
        for source_node, target_node in self.edges:
            if source_node == node:
                children.add(target_node)
        return children

    def parents(self, node: T) -> set[T]:
        parents = set()
        for source_node, target_node in self.edges:
            if target_node == node:
                parents.add(source_node)
        return parents


def add_domain_group_in_dependency_graph(
    graph: Graph[frozenset[str]], domain_group: frozenset[str]
) -> None:
    if domain_group in graph.nodes:
        return
    graph.add_node(domain_group)
    for existing_group in graph.nodes:
        # if domain_group is a subsset of existing_group
        if domain_group < existing_group:
            graph.add_edge((existing_group, domain_group))
        elif existing_group < domain_group:
            graph.add_edge((domain_group, existing_group))


def build_dependency_graph(
    domain_groups: list[frozenset[str]],
) -> Graph[frozenset[str]]:
    graph: Graph[frozenset[str]] = Graph()
    for domain_group in domain_groups:
        add_domain_group_in_dependency_graph(graph, domain_group)
    return graph
