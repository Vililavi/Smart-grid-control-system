from copy import deepcopy
from math import sqrt, log2
from dataclasses import dataclass, field
from itertools import count
from random import choice, random, gauss
from typing import Tuple, Optional

from neat.genetics.genes import NodeGene, ConnectionGene, NodeType


@dataclass(slots=True)
class WeightOptions:
    init_mean: float
    init_stdev: float
    max_adjust: float
    min_val: float
    max_val: float

    def get_new_val(self) -> float:
        return gauss(self.init_mean, self.init_stdev)

    def adjust(self, old_val: float) -> float:
        change = 2 * self.max_adjust * random() - self.max_adjust
        new_val = old_val + change
        return min(self.max_val, max(self.min_val, new_val))


@dataclass(slots=True)
class MutationParams:
    add_node_prob: float
    add_connection_prob: float
    adjust_weight_prob: float
    replace_weight_prob: float
    adjust_bias_prob: float
    replace_bial_prob: float
    weight_options: WeightOptions
    bias_options: WeightOptions


@dataclass(slots=True)
class Innovations:
    """Used to track the innovations of the current generation during reproduction (mutation) process."""
    split_connections: dict[tuple[int, int], int] = field(init=False, default_factory=lambda: {})
    added_connections: dict[tuple[int, int], int] = field(init=False, default_factory=lambda: {})


@dataclass(slots=True)
class Genome:
    """Genetic representation of a neural network."""
    key: int
    inputs: dict[int, NodeGene]
    output_keys: list[int]
    nodes: dict[int, NodeGene]  # Nodes chromosome
    connections: dict[(int, int), ConnectionGene]  # Connections chromosome
    conns_by_innovation: dict[int, ConnectionGene] = field(init=False)
    fitness: Optional[float] = None

    def __post_init__(self):
        self.conns_by_innovation = {}
        for conn in self.connections.values():
            self.conns_by_innovation[conn.innovation_num] = conn

    @classmethod
    def create_new(
        cls,
        key: int,
        num_inputs: int,
        num_outputs: int,
        weight_options: WeightOptions,
        bias_options: WeightOptions,
        node_start: int = 0,
        conn_start: int = 1
    ) -> "Genome":
        """Create a new Genome with random weights and without hidden nodes."""
        output_keys = [i for i in range(node_start + num_inputs, node_start + num_inputs + num_outputs)]
        inputs = {}
        nodes = {}
        connections = {}
        for i in range(num_inputs):
            inputs[node_start + i] = NodeGene(i, NodeType.SENSOR, bias_options.get_new_val())
            for j in output_keys:
                c_key = (i, j)
                connections[c_key] = ConnectionGene(
                    i, j, weight_options.get_new_val(), True, conn_start + i * j
                )
        for i in output_keys:
            nodes[i] = NodeGene(i, NodeType.OUTPUT, bias_options.get_new_val())
        return cls(key, inputs, output_keys, nodes, connections)

    @classmethod
    def from_crossover(cls, key: int, genome_1: "Genome", genome_2: "Genome", keep_disable_prob: float) -> "Genome":
        """Produces a new Genome (offspring) via crossover from two parent genomes."""
        if genome_1.fitness > genome_2.fitness:
            parent_1, parent_2 = genome_1, genome_2
        else:
            parent_1, parent_2 = genome_2, genome_1

        connections = Genome._get_inherited_connections(parent_1, parent_2, keep_disable_prob)
        nodes = {}
        for (in_key, out_key) in connections:
            if in_key in parent_1.nodes and in_key not in nodes:
                nodes[in_key] = deepcopy(parent_1.nodes[in_key])
            if out_key in parent_1.nodes and out_key not in nodes:
                nodes[out_key] = deepcopy(parent_1.nodes[out_key])
        return cls(key, parent_1.inputs, parent_1.output_keys, nodes, connections)

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

    def mutate(
        self,
        mutation_params: MutationParams,
        node_counter: count,
        conn_counter: count,
        innovations_in_curr_generation: Innovations
    ) -> None:
        """Mutates this genome"""
        if random() < mutation_params.add_node_prob:
            self._mutate_add_node(node_counter, conn_counter, innovations_in_curr_generation, mutation_params)
        if random() < mutation_params.add_connection_prob:
            self._mutate_add_connection(conn_counter, innovations_in_curr_generation, mutation_params.weight_options)
        self._mutate_weights(
            mutation_params.adjust_weight_prob, mutation_params.replace_weight_prob, mutation_params.weight_options
        )
        self._mutate_biases(
            mutation_params.adjust_bias_prob, mutation_params.replace_bial_prob, mutation_params.bias_options
        )

    def _mutate_add_node(
        self, node_counter: count, conn_counter: count, inns_in_curr_gen: Innovations, mutation_params: MutationParams
    ) -> Tuple[ConnectionGene, ConnectionGene]:
        """Mutates this genome by adding a node."""
        conn_to_split = choice(list(self.connections.values()))
        conn_to_split.enabled = False

        new_node_idx = self._add_node(conn_to_split, node_counter, inns_in_curr_gen, mutation_params.bias_options)

        c1 = self._add_connection(
            conn_to_split.node_in_idx, new_node_idx, 1.0, True, conn_counter, inns_in_curr_gen
        )
        c2 = self._add_connection(
            new_node_idx, conn_to_split.node_out_idx, conn_to_split.weight, True, conn_counter, inns_in_curr_gen
        )
        return c1, c2

    def _add_node(
        self,
        conn_to_split: ConnectionGene,
        node_counter: count,
        inns_in_curr_gen: Innovations,
        bias_options: WeightOptions,
    ) -> int:
        key = (conn_to_split.node_in_idx, conn_to_split.node_out_idx)
        if key in inns_in_curr_gen.split_connections:
            new_node_idx = inns_in_curr_gen.split_connections[key]
        else:
            new_node_idx = next(node_counter)
            inns_in_curr_gen.split_connections[key] = new_node_idx
        self.nodes[new_node_idx] = NodeGene(new_node_idx, NodeType.HIDDEN, bias_options.get_new_val())
        return new_node_idx

    def _mutate_add_connection(
        self, conn_counter: count, inns_in_curr_gen: Innovations, weight_options: WeightOptions
    ) -> Optional[ConnectionGene]:
        """Mutates this genome by adding a new connection."""
        possible_inputs = list(self.nodes.keys())
        possible_inputs.extend(list(self.inputs.keys()))
        in_key = choice(possible_inputs)
        out_key = choice(list(self.nodes.keys()))
        key = (in_key, out_key)
        if key in self.connections:
            self.connections[key].enabled = True
            return
        if in_key not in self.inputs:
            if self.nodes[in_key].node_type == NodeType.OUTPUT and self.nodes[out_key].node_type == NodeType.OUTPUT:
                return
        return self._add_connection(in_key, out_key, weight_options.get_new_val(), True, conn_counter, inns_in_curr_gen)

    def _add_connection(
        self,
        in_key: int,
        out_key: int,
        weight: float,
        enabled: bool,
        conn_counter: count,
        inns_in_curr_gen: Innovations
    ) -> ConnectionGene:
        assert out_key >= 0
        key = (in_key, out_key)
        if key in inns_in_curr_gen.added_connections:
            innov_num = inns_in_curr_gen.added_connections[key]
        else:
            innov_num = next(conn_counter)
            inns_in_curr_gen.added_connections[key] = innov_num
        connection = ConnectionGene(in_key, out_key, weight, enabled, innov_num)
        self.connections[key] = connection
        self.conns_by_innovation[innov_num] = connection
        return connection

    def _mutate_weights(self, adjust_prob: float, replace_prob: float, options: WeightOptions) -> None:
        for connection in self.connections.values():
            rand = random()
            if rand < replace_prob:
                connection.weight = options.get_new_val()
            elif rand < adjust_prob + replace_prob:
                connection.weight = options.adjust(connection.weight)

    def _mutate_biases(self, adjust_prob: float, replace_prob: float, options: WeightOptions) -> None:
        for node in self.nodes.values():
            rand = random()
            if rand < replace_prob:
                node.bias = options.get_new_val()
            elif rand < adjust_prob + replace_prob:
                node.bias = options.adjust(node.bias)

    @staticmethod
    def genome_distance(genome_1: "Genome", genome_2: "Genome", disjoint_coeff: float, weight_coeff: float) -> float:
        """Compute the distance of the two given genomes."""

        node_distance = Genome._compute_node_distance(genome_1, genome_2, disjoint_coeff)
        connection_distance = Genome._compute_connection_distance(genome_1, genome_2, disjoint_coeff, weight_coeff)
        return node_distance + connection_distance

    @staticmethod
    def _compute_node_distance(genome_1: "Genome", genome_2: "Genome", disjoint_coeff: float) -> float:
        if not (genome_1.nodes or genome_2.nodes):
            return 0.0
        disjoint_nodes = 0
        for key_1 in genome_1.nodes:
            if key_1 not in genome_2.nodes:
                disjoint_nodes += 1
        for key_2 in genome_2.nodes:
            if key_2 not in genome_1.nodes:
                disjoint_nodes += 1
        max_nodes = max(len(genome_1.nodes), len(genome_2.nodes))
        return disjoint_coeff * disjoint_nodes / max(1.0, log2(max_nodes))

    @staticmethod
    def _compute_connection_distance(
        genome_1: "Genome", genome_2: "Genome", disjoint_coeff: float, weight_coeff: float
    ) -> float:
        if not (genome_1.connections or genome_2.connections):
            return 0.0
        weight_diff = 0.0
        matching_connections = 0
        disjoint_connections = 0
        for key_1, conn_1 in genome_1.conns_by_innovation.items():
            conn_2 = genome_2.conns_by_innovation.get(key_1)
            if conn_2 is None:
                disjoint_connections += 1
            else:
                matching_connections += 1
                weight_diff += ConnectionGene.distance(conn_1, conn_2)
        for key_2 in genome_2.conns_by_innovation:
            if key_2 not in genome_1.conns_by_innovation:
                disjoint_connections += 1

        max_conn = max(len(genome_1.connections), len(genome_2.connections))
        disjoint_dist = disjoint_coeff * disjoint_connections / max(1.0, log2(max_conn))
        weight_dist = weight_coeff * weight_diff / matching_connections
        return disjoint_dist + weight_dist
