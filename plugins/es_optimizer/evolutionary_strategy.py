from typing import List

import numpy as np
from celery.utils.log import get_task_logger
from pymcdm.methods.mcda_method import MCDA_method

from plugins.es_optimizer.objective_functions import objective_function_array
from plugins.es_optimizer.weights import NormalizedWeights, Weights

TASK_LOGGER = get_task_logger(__name__)


def evolutionary_strategy(
    mcda: MCDA_method, metrics: List[np.ndarray], histogram_intersection: List[np.ndarray],
    is_cost: np.ndarray) -> NormalizedWeights:
    population_size = 20
    reproduction_factor = 4
    mutation_factor = 0.05
    metrics_cnt = metrics[0].shape[1]
    weights = np.random.random((population_size, metrics_cnt))

    for i in range(100):
        obj_values = objective_function_array(mcda, metrics[0], histogram_intersection[0], weights, is_cost)

        for m, hi in zip(metrics[1:], histogram_intersection[1:]):
            obj_values += objective_function_array(mcda, m, hi, weights, is_cost)

        obj_values /= len(metrics)

        sorted_indices = np.argsort(obj_values)

        obj_values = obj_values[sorted_indices]
        weights = weights[sorted_indices]

        TASK_LOGGER.info(obj_values[0])

        # remove worst weights
        weights: np.ndarray = weights[0:population_size // reproduction_factor]

        # clone and mutate weights
        new_weights = [weights]

        for i in range(reproduction_factor - 1):
            new_weights.append(weights.copy() + np.random.normal(scale=mutation_factor, size=weights.shape))

        weights = np.concatenate(new_weights, axis=0)

    best_weights = Weights.normalize(weights[0])

    return best_weights
