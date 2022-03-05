import math
from typing import List, Tuple

import numpy as np
from celery.utils.log import get_task_logger
from pymcdm.methods.mcda_method import MCDA_method
from sklearn import preprocessing

from plugins.es_optimizer.objective_functions import objective_function_array
from plugins.es_optimizer.preprocessing import normalize_weights


def roulette_wheel_selection(weights: np.ndarray, fitness: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    zero_min_fitness = fitness - np.nanmin(fitness)
    probability: np.ndarray = zero_min_fitness / np.nansum(zero_min_fitness)

    probability[np.isnan(probability)] = 0

    return rng.choice(weights, size=1, p=probability).reshape((-1))


def swap_gene_inplace(chromosome: np.ndarray, rng: np.random.Generator):
    index1 = rng.integers(0, chromosome.shape[0])
    index2 = rng.integers(0, chromosome.shape[0])

    tmp = chromosome[index1]
    chromosome[index1] = chromosome[index2]
    chromosome[index2] = tmp


def crossover(chromosome1: np.ndarray, chromosome2: np.ndarray, rng: np.random.Generator) -> Tuple[
    np.ndarray, np.ndarray]:
    index = rng.integers(0, chromosome1.shape[0])
    new_chromosome1 = np.concatenate([chromosome1[0:index], chromosome2[index:]])
    new_chromosome2 = np.concatenate([chromosome2[0:index], chromosome1[index:]])

    return new_chromosome1, new_chromosome2


TASK_LOGGER = get_task_logger(__name__)


def standard_genetic_algorithm(
    mcda: MCDA_method, metrics: List[np.ndarray], histogram_intersection: List[np.ndarray],
    is_cost: np.ndarray) -> np.ndarray:
    population_size = 100
    mutation_rate = 0.2
    crossover_rate = 0.85
    elitism_number = 5
    epochs = 200
    metrics_cnt = metrics[0].shape[1]
    weights = np.random.random((population_size, metrics_cnt))

    for i in range(epochs):
        fitness = -objective_function_array(mcda, metrics[0], histogram_intersection[0], weights, is_cost)

        for m, hi in zip(metrics[1:], histogram_intersection[1:]):
            fitness -= objective_function_array(mcda, m, hi, weights, is_cost)

        fitness /= len(metrics)

        sorted_indices = np.argsort(-fitness)

        fitness = fitness[sorted_indices]
        weights = weights[sorted_indices]

        TASK_LOGGER.info(fitness[0])

        new_weights = []
        rng = np.random.default_rng()

        # add elites
        for j in range(elitism_number):
            new_weights.append(weights[j])

        # add new weights through crossover
        crossover_cnt = math.floor(population_size * crossover_rate)

        for j in range(crossover_cnt // 2):
            new_weights.extend(
                crossover(
                    roulette_wheel_selection(weights, fitness, rng),
                    roulette_wheel_selection(weights, fitness, rng),
                    rng))

        # fill up with random chromosomes
        fill_up_cnt = population_size - len(new_weights)

        for j in range(fill_up_cnt):
            new_weights.append(roulette_wheel_selection(weights, fitness, rng))

        # mutate chromosomes
        mutations_cnt = math.floor((population_size - elitism_number) * metrics_cnt * mutation_rate * 0.5)

        for j in range(mutations_cnt):
            random_weights_index = rng.integers(elitism_number, population_size)  # elites will not be mutated
            swap_gene_inplace(new_weights[random_weights_index], rng)

        weights = np.stack(new_weights)

    best_weights = normalize_weights(weights[0])

    return best_weights
