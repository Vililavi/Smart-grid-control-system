from typing import Optional

import matplotlib.figure
from matplotlib import pyplot as plt
from neat.genetics.genome import Genome


def draw_species_graph(species_data: list[dict[int, tuple[list[Genome], int, Optional[float]]]]):
    species_fitnesses = {}
    top_member_fitnesses = {}
    species_sizes = {}
    for i, data_dict in enumerate(species_data):
        for idx, (members, _created, _fitness) in data_dict.items():
            if idx not in species_fitnesses:
                species_fitnesses[idx] = [] if i == 0 else [0.0] * i
                top_member_fitnesses[idx] = [] if i == 0 else [0.0] * i
                species_sizes[idx] = [] if i == 0 else [0] * i
            # species_fitnesses[idx].append(fitness)
            species_fitnesses[idx].append(sum([g.fitness for g in members]) / len(members))
            top_member_fitnesses[idx].append(max([g.fitness for g in members]))
            species_sizes[idx].append(len(members))

    for fitnesses in species_fitnesses.values():
        if len(fitnesses) < len(species_data):
            end_padding = [0.0] * (len(species_data) - len(fitnesses))
            fitnesses.extend(end_padding)

    for fitnesses in top_member_fitnesses.values():
        if len(fitnesses) < len(species_data):
            end_padding = [0.0] * (len(species_data) - len(fitnesses))
            fitnesses.extend(end_padding)

    for sizes in species_sizes.values():
        if len(sizes) < len(species_data):
            end_padding = [0] * (len(species_data) - len(sizes))
            sizes.extend(end_padding)

    fig: matplotlib.figure.Figure = plt.figure("Species fitnesses", figsize=(10.0, 8.0))
    for idx, fitnesses in species_fitnesses.items():
        plt.plot(fitnesses, label=f"Species {idx}")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower left")
    plt.show()

    fig: matplotlib.figure.Figure = plt.figure("Top species member fitnesses", figsize=(10.0, 8.0))
    for idx, fitnesses in top_member_fitnesses.items():
        plt.plot(fitnesses, label=f"Species {idx}")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower left")
    plt.show()

    fig: matplotlib.figure.Figure = plt.figure("Species sizes", figsize=(10.0, 8.0))
    for idx, sizes in species_sizes.items():
        plt.plot(sizes, label=f"Species {idx}")
    plt.xlabel("Generation")
    plt.ylabel("Size")
    plt.legend(loc="lower left")
    plt.show()
