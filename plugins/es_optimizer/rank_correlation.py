import math

import numpy as np


def spearman(rank1: np.ndarray, rank2: np.ndarray):
    if (not np.issubdtype(rank1.dtype, np.integer)) or (not np.issubdtype(rank2.dtype, np.integer)):
        raise ValueError("rank arrays must be of integer type")

    if rank1.ndim > 1 or rank2.ndim > 1:
        raise ValueError("rank arrays must be of dimension 1")

    distance = rank1 - rank2
    size = rank1.shape[0]

    return 1 - (6 * np.sum(distance * distance)) / (size * (size * size - 1))


def _sign(value: float) -> float:
    return math.copysign(1.0, value)


def kendall(rank1: np.ndarray, rank2: np.ndarray):
    if (not np.issubdtype(rank1.dtype, np.integer)) or (not np.issubdtype(rank2.dtype, np.integer)):
        raise ValueError("rank arrays must be of integer type")

    if rank1.ndim > 1 or rank2.ndim > 1:
        raise ValueError("rank arrays must be of dimension 1")

    concordant_sum = 0
    size = rank1.shape[0]

    for i in range(size):
        for j in range(size):
            if i < j:
                concordant_sum += _sign(rank1[i] - rank1[j]) * _sign(rank2[i] - rank2[j])

    return 2.0 / (size * (size - 1)) * concordant_sum


def _test():
    rank1 = np.array([0, 1, 2, 3])
    rank2 = np.array([0, 2, 1, 3])

    print(spearman(rank1, rank2))
    print(kendall(rank1, rank2))


if __name__ == "__main__":
    _test()
