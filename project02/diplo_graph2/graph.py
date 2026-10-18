import csv
from pathlib import Path

class Node:
    def __init__(self, node_type: str, label: str, attributes: dict[str, str] = {},coord_in_case: tuple[int, int] = (-1, -1) , pos_in_case_text: int = -1):
        self.node_type = node_type
        self.label = label
        self.attributes = attributes
        self.coord_in_case = coord_in_case
        self.pos_in_case_text: int = pos_in_case_text

    def __str__(self):
        return f"Node(type={self.node_type}, label=\"{self.label}\", attributes={self.attributes}, pos_in_case={self.pos_in_case_text})"

class Edge:
    def __init__(self, from_node: int, to_node: int, relation: str, attributes: dict[str, str]):
        self.from_node = from_node
        self.to_node = to_node
        self.relation = relation
        self.attributes = attributes

    def __str__(self):
        return f"Edge(from={self.from_node}, to={self.to_node}, relation={self.relation}, attributes={self.attributes})"

class Graph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node: Node):
        self.nodes.append(node)

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def add_node_c(self, node_type: str, label: str, attributes: dict[str, str]) -> int:
        self.nodes.append(Node(node_type, label, attributes))
        return len(self.nodes) - 1

    def add_edge_c(self, from_node: int, to_node: int, relation: str, attributes: dict[str, str]) -> int:
        if 0 <= from_node < len(self.nodes) and 0 <= to_node < len(self.nodes):
            self.edges.append(Edge(from_node, to_node, relation, attributes))
            return len(self.edges) - 1
        else:
            raise ValueError("Both nodes must exist in the graph before adding an edge.")

    def node(self, index: int) -> Node:
        if 0 <= index < len(self.nodes):
            return self.nodes[index]
        else:
            raise IndexError("Node index out of range.")

    def first_node_with_label(self, label: str) -> Node:
        for node in self.nodes:
            if node.label == label:
                return node
        raise ValueError(f"No node found with label: {label}")

    def edge(self, index: int) -> Edge:
        if 0 <= index < len(self.edges):
            return self.edges[index]
        else:
            raise IndexError("Edge index out of range.")

    def get_nodes_by_type(self, node_type: str) -> list[Node]:
        return [node for node in self.nodes if node.node_type == node_type]

    def get_edges_by_relation(self, relation: str) -> list[Edge]:
        return [edge for edge in self.edges if edge.relation == relation]

    def len_nodes(self) -> int:
        return len(self.nodes)

    def len_edges(self) -> int:
        return len(self.edges)

    def save_to_csvs(self, nodes_file: str | Path, edges_file: str | Path):
        """Salva os nós e arestas do grafo em arquivos CSV nos formatos esperados pelo projeto."""
        nodes_path = Path(nodes_file)
        edges_path = Path(edges_file)
        nodes_path.parent.mkdir(parents=True, exist_ok=True)
        edges_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Salva o CSV de Nós
        with nodes_path.open("w", newline="", encoding="utf-8") as f_n:
            writer = csv.DictWriter(
                f_n,
                fieldnames=["node_index", "node_type", "label", "attributes", "coord_in_case", "pos_in_case_text"],
            )
            writer.writeheader()
            for idx, n in enumerate(self.nodes):
                attr_str = "; ".join(f"{k}={v}" for k, v in (n.attributes or {}).items() if v != "")
                writer.writerow({
                    "node_index": idx,
                    "node_type": n.node_type,
                    "label": n.label,
                    "attributes": attr_str,
                    "coord_in_case": f"({n.coord_in_case[0]}, {n.coord_in_case[1]})",
                    "pos_in_case_text": n.pos_in_case_text,
                })

        # 2. Salva o CSV de Arestas
        with edges_path.open("w", newline="", encoding="utf-8") as f_e:
            writer = csv.DictWriter(
                f_e,
                fieldnames=["edge_index", "from_node", "to_node", "source_label", "target_label", "relation", "attributes"],
            )
            writer.writeheader()
            for idx, e in enumerate(self.edges):
                attr_str = "; ".join(f"{k}={v}" for k, v in (e.attributes or {}).items() if v != "")
                src_label = self.node(e.from_node).label if 0 <= e.from_node < self.len_nodes() else ""
                tgt_label = self.node(e.to_node).label if 0 <= e.to_node < self.len_nodes() else ""
                writer.writerow({
                    "edge_index": idx,
                    "from_node": e.from_node,
                    "to_node": e.to_node,
                    "source_label": src_label,
                    "target_label": tgt_label,
                    "relation": e.relation,
                    "attributes": attr_str,
                })

    def __str__(self):
        return "-------- Nodes --------\n" +\
        "\n".join(f"{i}: {node}" for i, node in enumerate(self.nodes)) +\
        "\n\n-------- Edges --------\n" +\
        "\n".join(f"{i}: {edge}" for i, edge in enumerate(self.edges))

def main():
    print("Testando uso do grafo")
    graph = Graph()
    node1_index = graph.add_node_c("TypeA", "Node1", {"attr1": "value1"})
    node2_index = graph.add_node_c("TypeB", "Node2", {"attr2": "value2"})
    edge_index = graph.add_edge_c(node1_index, node2_index, "RelationX", {"weight": "5"})

    print(graph)

if __name__ == "__main__":
    main()