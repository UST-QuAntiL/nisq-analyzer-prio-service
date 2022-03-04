import numpy as np


def convert_scores_to_ranking(scores: np.ndarray, higher_scores_are_better: bool) -> np.ndarray:
    if higher_scores_are_better:
        scores = -scores

    return np.argsort(np.argsort(scores))


def sort_array_with_ranking(array: np.ndarray, ranking: np.ndarray) -> np.ndarray:
    return array[np.argsort(ranking)]


def _test():
    test = np.array([3, 4, 5, 2])
    ranking = convert_scores_to_ranking(test, False)
    test_sorted = sort_array_with_ranking(test, ranking)
    print(ranking)
    print(test_sorted)


if __name__ == "__main__":
    _test()
