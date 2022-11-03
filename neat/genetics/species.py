from dataclasses import dataclass, field
from typing import Optional
from itertools import count

from genome import Genome


@dataclass
class Species:
    """Keeps track of genomes in a single species."""

    key: int
    created: int
    last_improved: int
    representative: Optional[Genome] = None
    members: dict[int, Genome] = field(default_factory=lambda: {})
    fitness: Optional[float] = None
    adjusted_fitness: Optional[float] = None
    fitness_history: list[float] = field(default_factory=lambda: [])

    def update(self, representative: Genome, members: dict[int, Genome]):
        """Set new representative and members"""
        self.representative = representative
        self.members = members

    def get_fitnesses(self) -> list[float]:
        """Returns fitnesses of this species' members as a list."""
        return [m.fitness for m in self.members.values()]


class DistanceCache:
    """Caches genome distances for purposes of speciation."""

    def __init__(self):
        self.distances: dict[tuple[int, int], float] = {}

    def __call__(self, genome_1: Genome, genome_2: Genome) -> float:
        """Get distance of given genomes."""
        key_1 = genome_1.key
        key_2 = genome_2.key
        d = self.distances.get((key_1, key_2))
        if d is None:
            d = Genome.genome_distance(genome_1, genome_2)
            self.distances[key_1, key_2] = d
            self.distances[key_2, key_1] = d
        return d


class SpeciesSet:
    """Handles speciation, i.e. the division of the population into species."""

    def __init__(self):
        self._indexer = count(1)
        self._species: dict[int, Species] = {}
        self._genome_to_species: dict[int, int] = {}

    def get_species_id(self, genome_id: int) -> int:
        """Get id of the species for a given individual."""
        return self._genome_to_species[genome_id]

    def get_species(self, genome_id: int) -> Species:
        """Returns the species object of the given individual."""
        species_id = self._genome_to_species[genome_id]
        return self._species[species_id]

    def speciate(self, population: dict[int, Genome], generation: int) -> None:
        """Divide population into species."""

        # TODO
