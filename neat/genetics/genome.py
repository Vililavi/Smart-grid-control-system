from dataclasses import dataclass, field
from itertools import count
from random import choice, random
from typing import Tuple, Optional

from neat.genetics.genes import NodeGene, ConnectionGene, NodeType


@dataclass
class Genome:
    """Genetic representation of a neural network."""
    key: int
    non_input_keys: list[int]
    nodes: list[NodeGene]  # Nodes chromosome
    connections: dict[(int, int), ConnectionGene]  # Connections chromosome
    conns_by_innovation: dict[int, ConnectionGene] = field(init=False)
    fitness: Optional[float] = None

    def __post_init__(self):
        self.conns_by_innovation = {}
        for conn in self.connections.values():
            self.conns_by_innovation[conn.innovation_num] = conn

    def to_phenotype(self):
        """Retrieve a phenotype (neural network) described by this genome."""

        # TODO: Have this here or as a function in some other module?

    @classmethod
    def create_new(cls, key: int, num_inputs: int, num_outputs: int, innov_start: int) -> "Genome":
        """Create a new Genome with random weights and without hidden nodes."""
        non_input_keys = [num_inputs + i for i in range(num_outputs)]
        nodes = []
        connections = {}
        for i in range(num_inputs):
            nodes.append(NodeGene(i, NodeType.SENSOR))
            for j in range(num_outputs):
                c_key = (i, j)
                connections[c_key] = ConnectionGene(i, num_inputs + j, 2 * random() - 1, True, innov_start + i * j)
        for i in range(num_outputs):
            nodes.append(NodeGene(i + num_inputs, NodeType.OUTPUT))
        return Genome(key, non_input_keys, nodes, connections)

    @classmethod
    def from_crossover(cls, key: int, genome_1: "Genome", genome_2: "Genome", keep_disable_prob: float) -> "Genome":
        """Produces a new Genome (offspring) via crossover from two parent genomes."""
        if genome_1.fitness > genome_2.fitness:
            parent_1, parent_2 = genome_1, genome_2
        else:
            parent_1, parent_2 = genome_2, genome_1

        connections = Genome._get_inherited_connections(parent_1, parent_2, keep_disable_prob)
        return Genome(key, parent_1.non_input_keys, parent_1.nodes, connections)

    @staticmethod
    def _get_inherited_connections(
        parent_1: "Genome", parent_2: "Genome", keep_disable_prob: float
    ) -> dict[(int, int), ConnectionGene]:
        conns = {}
        for innov_num, conn in parent_1.conns_by_innovation.items():
            key = (conn.node_in_idx, conn.node_out_idx)
            if innov_num in parent_2.conns_by_innovation:
                conns[key] = conn.crossover(parent_2.conns_by_innovation[innov_num], keep_disable_prob)
            else:
                conns[key] = conn
        return conns

    def mutate(self, add_node_prob: float, add_conn_prob: float, innov_counter: count) -> None:
        """Mutates this genome"""
        # TODO: Move this for easier tracking of innovations in current generation
        if random() < add_node_prob:
            self.mutate_add_node(innov_counter)
        if random() < add_conn_prob:
            self.mutate_add_connection(innov_counter)

    def mutate_add_node(self, innov_counter: count) -> Tuple[ConnectionGene, ConnectionGene]:
        """Mutates this genome by adding a node."""
        new_node_idx = len(self.nodes)
        node = NodeGene(new_node_idx, NodeType.HIDDEN)
        self.nodes.append(node)
        self.non_input_keys.append(new_node_idx)

        conn_to_split = choice(list(self.connections.values()))
        conn_to_split.enabled = False
        c1 = self._add_connection(conn_to_split.node_in_idx, new_node_idx, 1.0, True, innov_counter)
        c2 = self._add_connection(new_node_idx, conn_to_split.node_out_idx, conn_to_split.weight, True, innov_counter)
        return c1, c2

    def mutate_add_connection(self, innov_counter: count) -> Optional[ConnectionGene]:
        """Mutates this genome by adding a new connection."""
        in_key = choice(self.nodes).idx
        out_key = choice(self.non_input_keys)
        key = (in_key, out_key)
        if key in self.connections:
            self.connections[key].enabled = True
            return
        if self.nodes[in_key].node_type == NodeType.OUTPUT and self.nodes[out_key].node_type == NodeType.OUTPUT:
            return
        return self._add_connection(in_key, out_key, 2 * random() - 1, True, innov_counter)

    def _add_connection(
        self, in_key: int, out_key: int, weight: float, enabled: bool, innov_counter: count
    ) -> ConnectionGene:
        assert out_key >= 0
        key = (in_key, out_key)
        connection = ConnectionGene(in_key, out_key, weight, enabled, next(innov_counter))
        self.connections[key] = connection
        return connection
