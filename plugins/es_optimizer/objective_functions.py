from typing import List

import numpy as np
from pymcdm.methods.mcda_method import MCDA_method


def objective_function(mcda: MCDA_method, metrics: np.ndarray, histogram_intersection: np.ndarray, weights: np.ndarray, is_cost: np.ndarray) -> float:
    target_scores = histogram_intersection
    scores = mcda(metrics, weights, is_cost)

    # normalization
    target_scores -= np.min(target_scores)
    target_scores /= np.max(target_scores)

    scores -= np.min(scores)
    scores /= np.max(scores)

    # mean square error
    loss = np.mean((target_scores - scores) * (target_scores - scores))

    return loss.item()


def objective_function_array(
    mcda: MCDA_method, metrics: np.ndarray, histogram_intersection: np.ndarray, weights: np.ndarray, is_cost: np.ndarray) -> np.ndarray:
    return np.array([objective_function(mcda, metrics, histogram_intersection, w, is_cost) for w in weights], dtype=float)


def objective_function_all_circuits(
    mcda: MCDA_method, weights: np.ndarray, metrics: List[np.ndarray], histogram_intersections: List[np.ndarray], is_cost: np.ndarray) -> float:
    error = 0.0

    for i in range(len(metrics)):
        error += objective_function(mcda, metrics[i], histogram_intersections[i], weights, is_cost)

    error = error / len(metrics)

    return error
