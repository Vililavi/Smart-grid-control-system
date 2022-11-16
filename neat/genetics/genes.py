from dataclasses import dataclass
from enum import Enum
from random import random


class NodeType(Enum):
    """
    Enum for keeping track of different node types.

    SENSOR nodes cannot have inputs added or removed.
    OUTPUT nodes cannot have their original output removed, but they can get new loop-back outputs
    (connections to the same or previous nodes).
    """
    SENSOR = 0
    OUTPUT = 1
    HIDDEN = 2


@dataclass(slots=True)
class NodeGene:
    """Gene representation of a node of a neural network."""
    idx: int
    node_type: NodeType
    bias: float


@dataclass(slots=True)
class ConnectionGene:
    """Gene representation of a connection in a network."""
    node_in_idx: int
    node_out_idx: int
    weight: float
    enabled: bool
    innovation_num: int

    def crossover(self, other_conn: "ConnectionGene",  keep_disable_prob: float) -> "ConnectionGene":
        """Crossover this connection gene with another one"""
        assert self.node_in_idx == other_conn.node_in_idx
        assert self.node_out_idx == other_conn.node_out_idx
        assert self.innovation_num == other_conn.innovation_num

        weight = other_conn.weight
        if random() < 0.5:
            weight = self.weight

        enabled = True
        if (not self.enabled or not other_conn.enabled) and random() < keep_disable_prob:
            enabled = False

        return ConnectionGene(self.node_in_idx, self.node_out_idx, weight, enabled, self.innovation_num)

    @staticmethod
    def distance(conn_1: "ConnectionGene", conn_2: "ConnectionGene") -> float:
        """Computes the distance of two ConnectionGenes"""
        assert conn_1.innovation_num == conn_2.innovation_num
        return abs(conn_1.weight - conn_2.weight)
