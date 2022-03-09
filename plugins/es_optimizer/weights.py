import numpy as np


class NormalizedWeights:
    def __init__(self, normalized_weights: np.ndarray):
        self.normalized_weights = normalized_weights


class Weights:
    @staticmethod
    def _logistic_function(values: np.ndarray, maximum_value: float, steepness: float, midpoint: float) -> np.ndarray:
        return maximum_value / (1 + np.exp(-steepness * (values - midpoint)))

    @staticmethod
    def normalize(weights: np.ndarray) -> NormalizedWeights:
        weights_range_zero_one = Weights._logistic_function(weights, 1, 5, 0.5)
        weights_sum_one = weights_range_zero_one / np.sum(weights_range_zero_one)

        return NormalizedWeights(weights_sum_one)
