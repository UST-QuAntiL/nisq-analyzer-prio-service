import math
from typing import List, Tuple

import numpy as np
from pymcdm.methods.mcda_method import MCDA_method

from . import rank_correlation
from .borda_count import borda_count_rank
from .experiments.tools.data_loader import is_cost
from .tools.ranking import create_mcda_ranking
from .weights import NormalizedWeights


def calculate_average_spearman(mcda: MCDA_method, metrics: List[np.ndarray], original_weights: NormalizedWeights, disturbed_weights: NormalizedWeights, rankings_for_borda: List[List[np.ndarray]] = None) -> float:
    average_spearman = 0

    for i, m in enumerate(metrics):
        original_ranking = create_mcda_ranking(mcda, m, original_weights, is_cost)
        disturbed_ranking = create_mcda_ranking(mcda, m, disturbed_weights, is_cost)

        if rankings_for_borda is not None and len(rankings_for_borda) > 0:
            original_ranking = borda_count_rank([original_ranking] + rankings_for_borda[i])
            disturbed_ranking = borda_count_rank([disturbed_ranking] + rankings_for_borda[i])

        spearman = rank_correlation.spearman(original_ranking, disturbed_ranking)
        average_spearman += spearman

    average_spearman /= len(metrics)

    return average_spearman


def find_changing_factors(mcda: MCDA_method, metrics: List[np.ndarray], original_weights: NormalizedWeights, rankings_for_borda: List[List[np.ndarray]] = None, step_size: float = 0.01, upper_bound: float = 145, lower_bound: float = 0.00657) -> Tuple[List[float], List[List[List[float]]], List[float], List[List[List[float]]]]:
    def find_factors(factor_adjustment: float, iterations: int) -> Tuple[List[float], List[List[List[float]]]]:
        changing_factors = []
        changed_rankings = []

        for i in range(original_weights.normalized_weights.shape[0]):
            factor = 1.0
            changing_factor = float("nan")
            changed_ranking: List[List[float]] = []

            for j in range(iterations):
                disturbed_weights: np.ndarray = original_weights.normalized_weights.copy()
                disturbed_weights[i] *= factor
                disturbed_weights /= np.sum(disturbed_weights)

                avg_spearman = calculate_average_spearman(
                    mcda, metrics, original_weights, NormalizedWeights(disturbed_weights), rankings_for_borda)

                if avg_spearman < 0.99999:
                    changing_factor = factor

                    for m in metrics:
                        disturbed_ranking: List[float] = create_mcda_ranking(mcda, m, NormalizedWeights(disturbed_weights), is_cost).tolist()
                        changed_ranking.append(disturbed_ranking)
                    break

                factor *= factor_adjustment

            changing_factors.append(changing_factor)
            changed_rankings.append(changed_ranking)

        return changing_factors, changed_rankings

    decreasing_iterations = math.floor(math.log10(lower_bound) / math.log(1.0 - step_size))
    increasing_iterations = math.floor(math.log10(upper_bound) / math.log(1.0 + step_size))

    decreasing_factors, decreasing_ranks = find_factors(1.0 - step_size, decreasing_iterations)
    increasing_factors, increasing_ranks = find_factors(1.0 + step_size, increasing_iterations)

    return decreasing_factors, decreasing_ranks, increasing_factors, increasing_ranks
