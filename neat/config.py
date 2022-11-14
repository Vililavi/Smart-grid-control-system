from dataclasses import dataclass


@dataclass
class NeatParams:
    """Dataclass for storing NEAT-specific parameters."""
    population_size: int

    repro_survival_rate: float
    min_species_size: int
    max_stagnation: int
    num_surviving_elite_species: int

    compatibility_threshold: float
    disjoint_coefficient: float
    weight_coefficient: float

    keep_disabled_probability: float
    node_mutation_probability: float
    connection_mutation_probability: float
