from typing import List

import numpy as np
from celery.utils.log import get_task_logger
from pymcdm.methods.mcda_method import MCDA_method
from sklearn import preprocessing

from plugins.es_optimizer.preprocessing import normalize_weights

TASK_LOGGER = get_task_logger(__name__)


def objective_function(mcda: MCDA_method, metrics: np.ndarray, histogram_intersection: np.ndarray, weights: np.ndarray, is_cost: np.ndarray) -> float:
    target_scores = histogram_intersection
    weights = normalize_weights(weights)
    scores = mcda(metrics, weights, is_cost)

    # normalization
    normalized_target_scores = target_scores - np.min(target_scores)
    normalized_target_scores /= np.max(normalized_target_scores)

    normalized_scores = scores - np.min(scores)
    normalized_scores /= np.max(normalized_scores)

    # mean square error
    loss = np.mean((normalized_target_scores - normalized_scores) * (normalized_target_scores - normalized_scores))

    return loss.item()


def objective_function_array(
    mcda: MCDA_method, metrics: np.ndarray, histogram_intersection: np.ndarray, weights: np.ndarray, is_cost: np.ndarray) -> np.ndarray:
    return np.array([objective_function(mcda, metrics, histogram_intersection, w, is_cost) for w in weights], dtype=float)


def objective_function_all_circuits(
    weights: np.ndarray, mcda: MCDA_method, metrics: List[np.ndarray], histogram_intersections: List[np.ndarray], is_cost: np.ndarray) -> float:
    error = 0.0

    for i in range(len(metrics)):
        error += objective_function(mcda, metrics[i], histogram_intersections[i], weights, is_cost)

    error = error / len(metrics)

    TASK_LOGGER.info(error)

    return error
