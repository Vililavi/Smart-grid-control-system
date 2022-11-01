from dataclasses import dataclass

from neat.genetics.genes import NodeGene, ConnectionGene


@dataclass
class Genome:
    """Genetic representation of a neural network."""
    key: int
    nodes: list[NodeGene]  # Nodes chromosome
    connections: list[ConnectionGene]  # Connections chromosome
    fitness: float

    def to_phenotype(self):
        """Retrieve a phenotype (neural network) described by this genome."""

        # TODO: Have this here or as a function in some other module?

    def mutate(self):
        """Mutate this genome"""
        # TODO: Is this needed or are the methods below enough (and this logic should be left to the mutation module)?

    def add_node(self):
        """Mutation by adding a node."""

        # TODO

    def add_connection(self):
        """Mutation by adding a new connection."""

        # TODO
