"""Coordinate and execute NEAT algorithm."""

from itertools import count
from typing import Optional, Callable

from neat.genetics.genome import Genome
from neat.genetics.species import SpeciesSet


class Evolution:
    """Tracks the evolution of a population of species and genomes."""

    def __init__(self, num_inputs: int, num_outputs: int, population_size: int):
        self.generation = 0
        self.innovation_counter = count(num_inputs * num_outputs + 1)
        self.population = self._get_initial_population(num_inputs, num_outputs, population_size)
        self.species_set = SpeciesSet()
        self.species_set.speciate(self.population, self.generation)
        self.best_genome: Optional[Genome] = None

    @staticmethod
    def _get_initial_population(num_inputs: int, num_outputs: int, population_size: int) -> dict[int, Genome]:
        genomes: dict[int, Genome] = {}
        for i in range(population_size):
            genomes[i] = Genome.create_new(i, num_inputs, num_outputs, 1)
        return genomes

    def run(self, fitness_function: Callable[[list[tuple[int, Genome]]], None], fitness_goal: float, n: int) -> Genome:
        for _ in range(n):
            fitness_function(list(self.population.items()))

            best = self._get_best_genome()
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            if self.best_genome.fitness > fitness_goal:
                break

            # TODO: create new generation
            #  - reproduction
            #  - mutation

            self.species_set.speciate(self.population, self.generation)
            self.generation += 1

    def _get_best_genome(self) -> Genome:
        best = None
        for genome in self.population.values():
            if genome.fitness is None:
                raise RuntimeError(f"Genome {genome} missing fitness value.")
            if best is None or genome.fitness > best.fitness:
                best = genome
        return best
