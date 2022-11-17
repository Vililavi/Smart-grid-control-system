from dataclasses import dataclass, field
from typing import Optional
from itertools import count

from neat.genetics.genome import Genome


@dataclass(slots=True)
class Species:
    """Keeps track of genomes in a single species."""

    key: int
    created: int
    last_improved: int = field(init=False)
    representative: Optional[Genome] = None
    members: dict[int, Genome] = field(default_factory=lambda: {})
    fitness: Optional[float] = None
    adjusted_fitness: Optional[float] = None
    fitness_history: list[float] = field(default_factory=lambda: [])

    def __post_init__(self):
        self.last_improved = self.created

    def update(self, representative: Genome, members: dict[int, Genome]):
        """Set new representative and members"""
        self.representative = representative
        self.members = members

    def get_fitnesses(self) -> list[float]:
        """Returns fitnesses of this species' members as a list."""
        return [m.fitness for m in self.members.values()]


class DistanceCache:
    """Caches genome distances for purposes of speciation."""
    __slots__ = ("disjoint_coeff", "weight_coeff", "distances")

    def __init__(self, disjoint_coefficient: float, weight_coefficient: float):
        self.disjoint_coeff = disjoint_coefficient
        self.weight_coeff = weight_coefficient
        self.distances: dict[tuple[int, int], float] = {}

    def __call__(self, genome_1: Genome, genome_2: Genome) -> float:
        """Get distance of given genomes."""
        key_1 = genome_1.key
        key_2 = genome_2.key
        d = self.distances.get((key_1, key_2))
        if d is None:
            d = Genome.genome_distance(genome_1, genome_2, self.disjoint_coeff, self.weight_coeff)
            self.distances[key_1, key_2] = d
            self.distances[key_2, key_1] = d
        return d


class SpeciesSet:
    """Handles speciation, i.e. the division of the population into species."""
    __slots__ = (
        "_indexer", "species", "_genome_to_species", "_compatibility_threshold", "disjoint_coeff", "weight_coeff"
    )

    def __init__(self, compatibility_threshold: float, disjoint_coefficient: float, weight_coefficient: float):
        self._indexer = count(1)
        self.species: dict[int, Species] = {}
        self._genome_to_species: dict[int, int] = {}
        self._compatibility_threshold = compatibility_threshold
        self.disjoint_coeff = disjoint_coefficient
        self.weight_coeff = weight_coefficient

    def get_species_id(self, genome_id: int) -> int:
        """Get id of the species for a given individual."""
        return self._genome_to_species[genome_id]

    def get_species(self, genome_id: int) -> Species:
        """Returns the species object of the given individual."""
        species_id = self._genome_to_species[genome_id]
        return self.species[species_id]

    def speciate(self, population: dict[int, Genome], generation: int) -> None:
        """Divide population into species."""
        unspeciated = set(population)
        distances = DistanceCache(self.disjoint_coeff, self.weight_coeff)
        new_representatives, new_members = self._get_new_representatives(unspeciated, population, distances)
        self._partition_to_species(unspeciated, population, new_representatives, new_members, distances)
        self._update_collections(population, new_representatives, new_members, generation)

    def _get_new_representatives(
        self, unspeciated: set[int], population: dict[int, Genome], distances: DistanceCache
    ) -> tuple[dict[int, int], dict[int, list[int]]]:
        """
        Finds new representatives for all species.
        New representative is the genome closest to the representative on the previous generation.
        """
        new_representatives = {}
        new_members = {}
        for species_id, species in self.species.items():
            candidates = []
            for genome_id in unspeciated:
                genome = population[genome_id]
                dist = distances(species.representative, genome)
                candidates.append((dist, genome))
            _, new_repr = min(candidates, key=lambda x: x[0])
            new_repr_id = new_repr.key
            new_representatives[species_id] = new_repr_id
            new_members[species_id] = [new_repr_id]
            unspeciated.remove(new_repr_id)
        return new_representatives, new_members

    def _partition_to_species(
        self,
        unspeciated: set[int],
        population: dict[int, Genome],
        representatives: dict[int, int],
        members: dict[int, list[int]],
        distances: DistanceCache
    ) -> None:
        while unspeciated:
            genome_id = unspeciated.pop()
            genome = population[genome_id]
            candidate_species = self._get_candidate_species(genome, population, representatives, distances)
            if candidate_species:
                _, species_id = min(candidate_species, key=lambda x: x[0])
                members[species_id].append(genome_id)
            else:
                species_id = next(self._indexer)
                representatives[species_id] = genome_id
                members[species_id] = [genome_id]

    def _get_candidate_species(
        self,
        genome: Genome,
        population: dict[int, Genome],
        representatives: dict[int, int],
        distances: DistanceCache
    ) -> list[tuple[float, int]]:
        candidates = []
        for species_id, repr_id in representatives.items():
            rep = population[repr_id]
            dist = distances(rep, genome)
            if dist < self._compatibility_threshold:
                candidates.append((dist, species_id))
        return candidates

    def _update_collections(
        self,
        population: dict[int, Genome],
        representatives: dict[int, int],
        members: dict[int, list[int]],
        generation: int
    ) -> None:
        self._genome_to_species = {}
        for species_id, repr_id in representatives.items():
            species = self.species.get(species_id)
            if species is None:
                species = Species(species_id, generation)
                self.species[species_id] = species

            s_members = members[species_id]
            for genome_id in s_members:
                self._genome_to_species[genome_id] = species_id

            s_members_dict = dict((genome_id, population[genome_id]) for genome_id in s_members)
            species.update(population[repr_id], s_members_dict)
