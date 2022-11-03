from neat.genetics.genome import Genome
from neat.genetics.species import SpeciesSet


class Generation:
    number: int
    best_genome: Genome


class Population:
    """Keeps track of all genes and species over the course of evolution."""
    genomes: dict[int, Genome]
    generation: Generation
    species_set: SpeciesSet
    best_genome: Genome

    # TODO: is this whole class needed???

    def update_best_genomes(self):
        pass

    def get_next_generation(self):
        # reproduce
        # mutate
        # speciate
        pass
