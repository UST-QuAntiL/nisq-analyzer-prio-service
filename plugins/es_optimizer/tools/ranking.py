import numpy as np
from pymcdm.methods import TOPSIS
from pymcdm.methods.mcda_method import MCDA_method

from plugins.es_optimizer.weights import NormalizedWeights


def convert_scores_to_ranking(scores: np.ndarray, higher_scores_are_better: bool) -> np.ndarray:
    if higher_scores_are_better:
        scores = -scores

    return np.argsort(np.argsort(scores))


def sort_array_with_ranking(array: np.ndarray, ranking: np.ndarray) -> np.ndarray:
    return array[np.argsort(ranking)]


def create_mcda_ranking(mcda: MCDA_method, metrics: np.ndarray, weights: NormalizedWeights, is_cost: np.ndarray) -> np.ndarray:
    return convert_scores_to_ranking(mcda(metrics, weights.normalized_weights, is_cost), True)


def _test_conversion():
    test = np.array([3, 4, 5, 2])
    ranking = convert_scores_to_ranking(test, False)
    test_sorted = sort_array_with_ranking(test, ranking)
    print(ranking)
    print(test_sorted)


def _test_mcda_ranking():
    mcda = TOPSIS()
    metrics = np.array([
        [5, 7],
        [7, 5]
    ])
    weights = np.array([0.2, 0.8])
    is_cost = np.array([1.0, -1.0])

    print(create_mcda_ranking(mcda, metrics, NormalizedWeights(weights), is_cost))


if __name__ == "__main__":
    # _test_mcda_ranking()
    original_ranking = np.array([1, 0, 3, 2])
    disturbed_ranking = np.array([3, 0, 1, 2])

    print(sort_array_with_ranking(disturbed_ranking, original_ranking))
