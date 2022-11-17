from typing import Optional
from matplotlib import pyplot as plt
from neat.genetics.genome import Genome


def draw_species_graph(species_data: list[dict[int, tuple[list[Genome], int, Optional[float]]]]):
    species_fitnesses = {}
    for i, data_dict in enumerate(species_data):
        for idx, (members, _created, _fitness) in data_dict.items():
            if idx not in species_fitnesses:
                species_fitnesses[idx] = [] if i == 0 else [0.0] * i
            # species_fitnesses[idx].append(fitness)
            species_fitnesses[idx].append(sum([g.fitness for g in members]) / len(members))
    for fitnesses in species_fitnesses.values():
        if len(fitnesses) < len(species_data):
            end_padding = [0.0] * (len(species_data) - len(fitnesses))
            fitnesses.extend(end_padding)

    fig = plt.figure("Species fitnesses", figsize=(10.0, 8.0))
    for idx, fitnesses in species_fitnesses.items():
        plt.plot(fitnesses, label=f"Species {idx}")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.show()
