import csv
from pathlib import Path

class Node:
    def __init__(self, node_type: str, label: str, attributes: dict[str, str] = {},coord_in_case: tuple[int, int] = (-1, -1) , pos_in_case_text: int = -1):
        self.node_type = node_type
        self.label = label
        self.attributes = attributes if attributes is not None else {}
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

    @staticmethod
    def _clean(v) -> str:
        # o parser do viewer separa atributos por ";" e linhas por quebra de linha
        return str(v).replace(";", ",").replace("\n", " ").strip()

    @staticmethod
    def _fmt_attrs(d: dict | None) -> str:
        return "; ".join(f"{k}={Graph._clean(v)}" for k, v in (d or {}).items() if v != "")

    def node_rows(self, id_prefix: str = "") -> list[dict]:
        return [
            {"node_id": f"{id_prefix}N{i}", "type": n.node_type,
                "label": n.label, "attributes": Graph._fmt_attrs(n.attributes)}
            for i, n in enumerate(self.nodes)
        ]

    def edge_rows(self, id_prefix: str = "") -> list[dict]:
        return [
            {"edge_id": f"{id_prefix}E{i}",
             "source_id": f"{id_prefix}N{e.from_node}",
             "target_id": f"{id_prefix}N{e.to_node}",
             "relation": e.relation, "attributes": Graph._fmt_attrs(e.attributes)}
            for i, e in enumerate(self.edges)
        ]

    @staticmethod
    def write_viewer_csvs(node_rows, edge_rows, nodes_file, edges_file):
        NODE_FIELDS = ["node_id", "type", "label", "attributes"]
        EDGE_FIELDS = ["edge_id", "source_id", "target_id", "relation", "attributes"]
        for path, fields, rows in ((nodes_file, NODE_FIELDS, node_rows), (edges_file, EDGE_FIELDS, edge_rows)):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)

    @staticmethod
    def save_many_graphs(graphs: dict[str, "Graph"], nodes_file, edges_file):
        """Junta vários casos em um único par de CSVs (IDs prefixados com o case_id)."""
        nodes, edges = [], []
        for case_id, g in graphs.items():
            nodes += g.node_rows(f"{case_id}:")
            edges += g.edge_rows(f"{case_id}:")
        Graph.write_viewer_csvs(nodes, edges, nodes_file, edges_file)

    def save_to_csvs(self, nodes_file, edges_file, id_prefix: str = ""):
        """CSVs no formato do Diplograph Viewer (node_id,type,label,attributes / edge_id,source_id,target_id,relation,attributes)."""
        Graph.write_viewer_csvs(self.node_rows(id_prefix), self.edge_rows(id_prefix), nodes_file, edges_file)

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