import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    normalized_weights = MinMaxScaler().fit_transform(weights.reshape((-1, 1))).reshape((-1))

    return normalized_weights / np.sum(normalized_weights)
