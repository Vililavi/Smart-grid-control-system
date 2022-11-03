from itertools import count

from neat.genetics.genome import Genome
from neat.genetics.species import SpeciesSet


class Reproduction:
    """Handles NEAT reproduction (creation of new genomes), including mutation of genomes."""

    def __init__(self, num_inputs: int, num_outputs: int):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.genome_indexer = count(1)
        self.innovation_counter = count(num_inputs * num_outputs + 1)  # TODO: have this here instead of in Evolution
        self.ancestors: dict[int, tuple[int, int]] = {}

    def create_new_population(self, population_size: int) -> dict[int, Genome]:
        """Creates an entirely new population with randomized minimal genomes."""
        genomes: dict[int, Genome] = {}
        for _ in range(population_size):
            key = next(self.genome_indexer)
            genomes[key] = Genome.create_new(key, self.num_inputs, self.num_outputs, 1)
            self.ancestors[key] = tuple()
        return genomes

    @staticmethod
    def _compute_spawn_amounts(
        adjusted_fitnesses: list[float], prev_sizes: list[int], pop_size: int, min_species_size: int
    ) -> list[int]:
        """Compute number of offspring for each species."""
        # TODO

    def reproduce(self, species: SpeciesSet, population_size: int, generation: int) -> dict[int, Genome]:
        """Create a new generation of genes via reproduction from the previous generation according to NEAT."""
        # TODO
