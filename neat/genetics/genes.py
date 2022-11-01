from dataclasses import dataclass
from enum import Enum


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


@dataclass
class NodeGene:
    """Gene representation of a node of a neural network."""
    idx: int
    node_type: NodeType


@dataclass
class ConnectionGene:
    """Gene representation of a connection in a network."""
    node_in_idx: int
    node_out_idx: int
    weight: float
    enabled: bool
    innovation_num: int
