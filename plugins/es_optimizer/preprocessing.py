import numpy as np


def _logistic_function(values: np.ndarray, maximum_value: float, steepness: float, midpoint: float) -> np.ndarray:
    return maximum_value / (1 + np.exp(-steepness * (values - midpoint)))


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights_max_one = weights / np.max(weights)
    weights_sum_one = weights_max_one / np.sum(weights_max_one)

    return weights_sum_one
