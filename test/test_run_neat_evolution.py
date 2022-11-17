from random import random

from neat.config import NeatParams
from neat.evolution import Evolution
from neat.genetics.genome import Genome


def mock_fitness_function(genomes: list[tuple[int, Genome]]) -> None:
    """
    A simple fitness function for demonstration purposes.
    Assigns a random fitness for each genome.

    If I understand correctly, the real fitness function should perform the training of the
    network derived from the genome and assign fitness based on the agent's performance.
    """
    for (_, genome) in genomes:
        genome.fitness = random()


def mock_species_fitness_function(species_fitnesses: list[float]) -> float:
    """A simple average for demonstration purposes"""
    return sum(species_fitnesses) / len(species_fitnesses)


def test_run_neat_evolution():
    """Test that the neat algorithm runs properly"""
    neat_config = NeatParams(
        population_size=20,

        repro_survival_rate=0.1,
        min_species_size=2,
        max_stagnation=5,
        num_surviving_elite_species=3,

        compatibility_threshold=3.0,
        disjoint_coefficient=1.0,
        weight_coefficient=0.3,
        keep_disabled_probability=0.5,

        node_mutation_probability=0.5,
        connection_mutation_probability=0.7,
        adjust_weight_prob=0.8,
        replace_weight_prob=0.1,
        adjust_bias_prob=0.6,
        replace_bial_prob=0.1,

        weight_init_mean=0.0,
        weight_init_stdev=1.0,
        weight_max_adjust=0.5,
        weight_min_val=-10.0,
        weight_max_val=10.0,

        bias_init_mean=0.0,
        bias_init_stdev=1.0,
        bias_max_adjust=0.5,
        bias_min_val=-10.0,
        bias_max_val=10.0,
    )
    evolution = Evolution(2, 3, neat_config, mock_species_fitness_function)
    evolution.run(mock_fitness_function, fitness_goal=2.0, n=10)


if __name__ == "__main__":
    test_run_neat_evolution()
