from math import inf
from multiprocessing import Pool
import time

import gym
import numpy as np
from numpy.typing import ArrayLike

import custom_envs.grid_v0

from visuals import draw_species_graph
from neat.config import NeatParams
from neat.evolution import Evolution
from neat.genetics.genome import Genome
from neat.nn.recurrent import RecurrentNetwork


def species_fitness_function(species_fitnesses: list[float]) -> float:
    return sum(species_fitnesses) / len(species_fitnesses)


def evaluate_network(network: RecurrentNetwork) -> float:
    num_days = 100
    env = gym.make("Grid-v0", max_total_steps=24 * num_days)

    num_episodes = 10
    total_reward = 0.0

    for episode in range(num_episodes):
        step_count = 0
        ep_reward = 0
        terminated = False
        state, _info = env.reset()

        while not terminated:
            state_list = _state_to_network_input(state)
            nn_output = network.activate(state_list)
            action = _network_output_to_action(nn_output)

            next_state, reward, terminated, _, _info = env.step(action)
            env.render()
            step_count += 1
            ep_reward += reward
            state = next_state

        total_reward += ep_reward / num_days
    return total_reward / num_episodes


def _state_to_network_input(state: tuple[ArrayLike, int, int]) -> list[float]:
    floats, price_counter, hour = state
    state_list = list(floats)
    state_list.extend([float(price_counter), float(hour)])

    # Normalize to [-1, 1]
    state_list[2] = (state_list[2] - 5.0) / 27.0
    state_list[3] = (state_list[3] - 900.0) / 900.0
    state_list[4] = (state_list[4] - 2999.0) / 2999.0
    state_list[5] = (state_list[5] - 0.7) / 0.7

    return state_list


def _network_output_to_action(nn_output: list[float]) -> ArrayLike:
    max_idx = 0
    max_val = -inf
    for i, val in enumerate(nn_output):
        if val > max_val:
            max_val = val
            max_idx = i
    tcl_action = max_idx // 20
    price_level = (max_idx - tcl_action * 20) // 5
    def_action = (max_idx - tcl_action * 20 - price_level * 4) // 2
    exc_action = (max_idx - tcl_action * 20 - price_level * 4) % 2
    action = np.array([tcl_action, price_level, def_action, exc_action], dtype=np.int64)
    return action


def evaluate_genome(idx_genome: tuple[int, Genome]) -> tuple[int, float]:
    idx, genome = idx_genome
    nn = RecurrentNetwork.create(genome)
    reward = evaluate_network(nn)
    genome.fitness = reward
    return idx, reward


def neat_fitness_function(genomes: list[tuple[int, Genome]]) -> None:
    with Pool() as pool:
        results = pool.map(evaluate_genome, genomes)

        for (idx_1, genome), (idx_2, fitness) in zip(genomes, results):
            assert idx_1 == idx_2, "Genomes or their order changed!"
            genome.fitness = fitness
        best_idx, fitness = max(results, key=lambda x: x[1])
        print(f"best genome: {best_idx}, fitness: {fitness}")


def main():
    neat_config = NeatParams(
        population_size=20,

        repro_survival_rate=0.1,    # What percentage of species' top members are used for reproduction
        min_species_size=2,
        max_stagnation=5,
        num_surviving_elite_species=3,  # Minimum number of species to be retained

        compatibility_threshold=0.3,
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
    evolution = Evolution(8, 4, neat_config, species_fitness_function)
    start_t = time.perf_counter()
    winning_genome = evolution.run(neat_fitness_function, fitness_goal=10.0, n=50)
    end_t = time.perf_counter()
    print(f"\nWinning genome: {winning_genome}\nFitness: {winning_genome.fitness}")
    print(f"total run time: {(end_t - start_t):.2f} seconds")
    draw_species_graph(evolution.species_history)


if __name__ == "__main__":
    main()
