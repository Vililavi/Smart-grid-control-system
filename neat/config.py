from dataclasses import dataclass


@dataclass(slots=True)
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
    adjust_weight_prob: float
    replace_weight_prob: float
    adjust_bias_prob: float
    replace_bial_prob: float

    weight_init_mean: float
    weight_init_stdev: float
    weight_max_adjust: float
    weight_min_val: float
    weight_max_val: float

    bias_init_mean: float
    bias_init_stdev: float
    bias_max_adjust: float
    bias_min_val: float
    bias_max_val: float
