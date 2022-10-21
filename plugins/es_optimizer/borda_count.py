from typing import List, Optional

import numpy as np

from .tools.ranking import convert_scores_to_ranking


def borda_count_rank(rankings: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    points = np.zeros_like(rankings[0], dtype=float)
    size = rankings[0].shape[0]

    if weights is None:
        weights = [1.0] * len(rankings)

    for ranking, weight in zip(rankings, weights):
        points += (size - ranking - 1) * weight

    return convert_scores_to_ranking(points, True)


def _test():
    rank1 = np.array([0, 1, 2, 3])
    rank2 = np.array([0, 1, 2, 3])
    rank3 = np.array([3, 0, 1, 2])

    print(borda_count_rank([rank1, rank2, rank3]))


def _test2():
    rank1 = np.array([0, 1, 2, 3])
    rank2 = np.array([3, 0, 1, 2])

    print(borda_count_rank([rank1, rank2], [0.0, 1.0]))


if __name__ == "__main__":
    _test2()
