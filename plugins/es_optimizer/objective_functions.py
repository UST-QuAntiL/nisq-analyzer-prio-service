from typing import List

import numpy as np
from celery.utils.log import get_task_logger
from pymcdm.methods.mcda_method import MCDA_method

from .weights import Weights

TASK_LOGGER = get_task_logger(__name__)


def objective_function(mcda: MCDA_method, metrics: np.ndarray, histogram_intersection: np.ndarray, weights: np.ndarray, is_cost: np.ndarray) -> float:
    target_scores = histogram_intersection
    scores = mcda(metrics, Weights.normalize(weights).normalized_weights, is_cost)

    # normalization
    normalized_target_scores = target_scores - np.min(target_scores)
    max_value = np.max(normalized_target_scores)

    if max_value == 0.0:
        # target_scores are all the same, we define the normalized values in this edge case as 0.5 for all elements
        normalized_target_scores = np.ones_like(normalized_target_scores) * 0.5
    else:
        normalized_target_scores /= max_value

    normalized_scores = scores - np.min(scores)
    max_value = np.max(normalized_scores)

    if max_value == 0.0:
        # scores are all the same, we define the normalized values in this edge case as 0.5 for all elements
        normalized_scores = np.ones_like(normalized_scores) * 0.5
    else:
        normalized_scores /= max_value

    # mean square error
    loss = np.mean((normalized_target_scores - normalized_scores) * (normalized_target_scores - normalized_scores))

    if np.isnan(loss.item()):
        TASK_LOGGER.error("loss is NaN")

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
