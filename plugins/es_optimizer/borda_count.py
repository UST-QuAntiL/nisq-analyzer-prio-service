from typing import List

import numpy as np


def borda_count(rankings: List[np.ndarray]) -> np.ndarray:
    points = np.zeros_like(rankings[0])
    size = rankings[0].shape[0]

    for ranking in rankings:
        points += size - ranking - 1

    return np.argsort(-points)


def _test():
    rank1 = np.array([0, 1, 2, 3])
    rank2 = np.array([0, 1, 2, 3])
    rank3 = np.array([3, 0, 1, 2])

    print(borda_count([rank1, rank2, rank3]))


if __name__ == "__main__":
    _test()
