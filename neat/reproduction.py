import sys
from itertools import count
from math import ceil
from random import choice
from typing import Callable

from neat.config import NeatParams
from neat.genetics.genome import Genome, Innovations, MutationParams, WeightOptions
from neat.genetics.species import SpeciesSet, Species


class Reproduction:
    """Handles NEAT reproduction (creation of new genomes), including mutation of genomes."""
    __slots__ = (
        "num_inputs",
        "num_outputs",
        "neat_params",
        "_weight_options",
        "_bias_options",
        "_mutate_params",
        "species_fitness_function",
        "genome_indexer",
        "node_counter",
        "conn_counter",
        "ancestors",
    )

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        neat_params: NeatParams,
        species_fitness_function: Callable[[list[float]], float]
    ):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.neat_params = neat_params
        self._weight_options = WeightOptions(
                neat_params.weight_init_mean,
                neat_params.weight_init_stdev,
                neat_params.weight_max_adjust,
                neat_params.weight_min_val,
                neat_params.weight_max_val,
            )
        self._bias_options = WeightOptions(
                neat_params.bias_init_mean,
                neat_params.bias_init_stdev,
                neat_params.bias_max_adjust,
                neat_params.bias_min_val,
                neat_params.bias_max_val,
            )
        self._mutate_params = MutationParams(
            neat_params.node_mutation_probability,
            neat_params.connection_mutation_probability,
            neat_params.adjust_weight_prob,
            neat_params.replace_weight_prob,
            neat_params.adjust_bias_prob,
            neat_params.replace_bial_prob,
            weight_options=self._weight_options,
            bias_options=self._bias_options,
        )
        self.species_fitness_function = species_fitness_function

        self.genome_indexer = count(1)
        self.node_counter = count(num_inputs + num_outputs)
        self.conn_counter = count(num_inputs * num_outputs)
        self.ancestors: dict[int, tuple[int, int]] = {}

    def create_new_population(self, population_size: int) -> dict[int, Genome]:
        """Creates an entirely new population with randomized minimal genomes."""
        genomes: dict[int, Genome] = {}
        for _ in range(population_size):
            key = next(self.genome_indexer)
            genomes[key] = Genome.create_new(
                key, self.num_inputs, self.num_outputs, self._weight_options, self._bias_options, 0
            )
            self.ancestors[key] = tuple()
        return genomes

    def reproduce(self, species_set: SpeciesSet, population_size: int, generation: int) -> dict[int, Genome]:
        """Create a new generation of genes via reproduction from the previous generation according to NEAT."""
        all_fitnesses = []
        surviving_species = []

        # NOTE: Comment out the next two lines if you don't want to use shared fitness within species.
        for species in species_set.species.values():
            self._adjust_genome_fitnesses_for_species(species)

        for _species_id, species, stagnant in self._get_stagnant_species(species_set, generation):
            if not stagnant:
                all_fitnesses.extend(genome.fitness for genome in species.members.values())
                surviving_species.append(species)

        adjusted_fitnesses = self._get_adjusted_fitnesses(all_fitnesses, surviving_species)
        previous_sizes = [len(s.members) for s in surviving_species]
        spawn_amounts = self._compute_spawn_amounts(
            adjusted_fitnesses, previous_sizes, population_size, self.neat_params.min_species_size
        )

        new_population = {}
        surviving_species_dict = {}
        new_innovations = Innovations()
        for spawn_amount, species in zip(spawn_amounts, surviving_species):
            possible_parents = self._select_genomes_for_reproduction(
                spawn_amount, species, surviving_species_dict, new_population
            )
            spawn_amount -= 1
            if spawn_amount <= 0:
                continue
            self._spawn_offspring(spawn_amount, possible_parents, new_population, new_innovations)
        species_set.species = surviving_species_dict
        return new_population

    @staticmethod
    def _adjust_genome_fitnesses_for_species(species: Species) -> None:
        for genome in species.members.values():
            genome.fitness = genome.fitness / len(species.members)

    def _get_stagnant_species(self, species_set: SpeciesSet, generation: int) -> list[tuple[int, Species, bool]]:
        """A species will die if it becomes stagnant (does not make progress) for long enough."""
        species_data = []
        for species_id, species in species_set.species.items():
            if species.fitness_history:
                prev_fitness = max(species.fitness_history)
            else:
                prev_fitness = -sys.float_info.max

            species.fitness = self.species_fitness_function(species.get_fitnesses())
            species.fitness_history.append(species.fitness)
            species.adjusted_fitness = None
            if prev_fitness is None or species.fitness > prev_fitness:
                species.last_improved = generation

            species_data.append((species_id, species))

        species_data.sort(key=lambda x: x[1].fitness)

        result = []
        num_non_stagnant = len(species_data)
        for idx, (species_id, species) in enumerate(species_data):

            stagnant_time = generation - species.last_improved
            is_stagnant = False
            if num_non_stagnant > self.neat_params.num_surviving_elite_species:
                is_stagnant = stagnant_time >= self.neat_params.max_stagnation

            if (len(species_data) - idx) <= self.neat_params.num_surviving_elite_species:
                is_stagnant = False

            if is_stagnant:
                num_non_stagnant -= 1

            result.append((species_id, species, is_stagnant))

        return result

    @staticmethod
    def _get_adjusted_fitnesses(fitnesses: list[float], remaining_species: list[Species]) -> list[float]:
        min_fitness = min(fitnesses)
        max_fitness = max(fitnesses)
        fitness_range = max(1.0, max_fitness - min_fitness)

        for species in remaining_species:
            species_fitnesses = species.get_fitnesses()
            mean_species_fitness = sum(species_fitnesses) / len(species_fitnesses)
            adjusted_fitness = (mean_species_fitness - min_fitness) / fitness_range
            species.adjusted_fitness = adjusted_fitness

        return [s.adjusted_fitness for s in remaining_species]

    @staticmethod
    def _compute_spawn_amounts(
        adjusted_fitnesses: list[float], prev_sizes: list[int], population_size: int, min_species_size: int
    ) -> list[int]:
        """Compute number of offspring for each species."""
        adj_fitness_sum = sum(adjusted_fitnesses)
        spawn_amounts = []
        for adj_fitness, prev_size in zip(adjusted_fitnesses, prev_sizes):
            if adj_fitness > 0:
                size = max(adj_fitness / adj_fitness_sum * population_size, min_species_size)
            else:
                size = min_species_size
            diff = (size - prev_size) * 0.5
            change = int(round(diff))
            spawn = prev_size
            if abs(change) > 0:
                spawn += change
            elif diff > 0:
                spawn += 1
            elif diff < 0:
                spawn -= 1
            spawn_amounts.append(spawn)

        total_spawn = sum(spawn_amounts)
        norm = population_size / total_spawn
        spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]

        return spawn_amounts

    def _select_genomes_for_reproduction(
        self,
        spawn_amount: int,
        species: Species,
        surviving_species: dict[int, Species],
        new_population: dict[int, Genome]
    ) -> list[tuple[int, Genome]]:
        assert spawn_amount > 0

        old_members = list(species.members.items())
        species.members = {}
        surviving_species[species.key] = species

        old_members.sort(reverse=True, key=lambda x: x[1].fitness)

        # keep the best genome
        key, elite = old_members[0]
        new_population[key] = elite

        repro_cutoff = int(ceil(self.neat_params.repro_survival_rate * len(old_members)))
        repro_cutoff = max(repro_cutoff, 2)
        return old_members[:repro_cutoff]

    def _spawn_offspring(
        self,
        spawn_amount: int,
        possible_parents: list[tuple[int, Genome]],
        new_population: dict[int, Genome],
        new_innovations: Innovations
    ) -> None:
        while spawn_amount > 0:
            spawn_amount -= 1

            parent_1_id, parent_1 = choice(possible_parents)
            parent_2_id, parent_2 = choice(possible_parents)

            genome_id = next(self.genome_indexer)
            offspring = Genome.from_crossover(
                genome_id, parent_1, parent_2, self.neat_params.keep_disabled_probability
            )
            offspring.mutate(
                self._mutate_params,
                self.node_counter,
                self.conn_counter,
                new_innovations
            )
            new_population[genome_id] = offspring
            self.ancestors[genome_id] = (parent_1_id, parent_2_id)
