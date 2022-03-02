import numpy as np


def spearman(rank1: np.ndarray, rank2: np.ndarray):
    if (not np.issubdtype(rank1.dtype, np.integer)) or (not np.issubdtype(rank2.dtype, np.integer)):
        raise ValueError("rank arrays must be of integer type")

    if rank1.ndim > 1 or rank2.ndim > 1:
        raise ValueError("rank arrays must be of dimension 1")

    distance = rank1 - rank2
    size = rank1.shape[0]

    return 1 - (6 * np.sum(distance * distance)) / (size * (size * size - 1))


def _test():
    print(spearman(np.array([0, 1, 2, 3]), np.array([0, 2, 1, 3])))


if __name__ == "__main__":
    _test()
