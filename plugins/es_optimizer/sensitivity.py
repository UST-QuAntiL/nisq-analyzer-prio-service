import math
from typing import List, Tuple, Optional

import numpy as np
from pymcdm.methods.mcda_method import MCDA_method

from . import rank_correlation
from .borda_count import borda_count_rank
from .tools.ranking import create_mcda_ranking
from .weights import NormalizedWeights


def calculate_average_spearman(mcda: MCDA_method, metrics: List[np.ndarray], is_cost: np.ndarray, original_weights: NormalizedWeights, disturbed_weights: NormalizedWeights, rankings_for_borda: List[List[np.ndarray]] = None, borda_count_weights: Optional[List[float]] = None) -> float:
    average_spearman = 0

    for i, m in enumerate(metrics):
        original_ranking = create_mcda_ranking(mcda, m, original_weights, is_cost)
        disturbed_ranking = create_mcda_ranking(mcda, m, disturbed_weights, is_cost)

        if rankings_for_borda is not None and len(rankings_for_borda) > 0:
            original_ranking = borda_count_rank([original_ranking] + rankings_for_borda[i], borda_count_weights)
            disturbed_ranking = borda_count_rank([disturbed_ranking] + rankings_for_borda[i], borda_count_weights)

        spearman = rank_correlation.spearman(original_ranking, disturbed_ranking)
        average_spearman += spearman

    average_spearman /= len(metrics)

    return average_spearman


def find_changing_factors(mcda: MCDA_method, metrics: List[np.ndarray], is_cost: np.ndarray, original_weights: NormalizedWeights, rankings_for_borda: List[List[np.ndarray]] = None, borda_count_weights: Optional[List[float]] = None, step_size: float = 0.01, upper_bound: float = 145, lower_bound: float = 0.00657) -> Tuple[List[float], List[List[List[int]]], List[List[List[int]]], List[float], List[List[List[int]]], List[List[List[int]]]]:
    def find_factors(factor_adjustment: float, iterations: int) -> Tuple[List[float], List[List[List[int]]], List[List[List[int]]]]:
        changing_factors = []
        changed_rankings = []
        changed_borda_rankings = []

        for i in range(original_weights.normalized_weights.shape[0]):
            factor = 1.0
            changing_factor = float("nan")
            changed_ranking: List[List[int]] = []
            changed_borda_ranking: List[List[int]] = []

            for j in range(iterations):
                disturbed_weights: np.ndarray = original_weights.normalized_weights.copy()
                disturbed_weights[i] *= factor
                disturbed_weights /= np.sum(disturbed_weights)

                avg_spearman = calculate_average_spearman(
                    mcda, metrics, is_cost, original_weights, NormalizedWeights(disturbed_weights), rankings_for_borda)

                if avg_spearman < 0.99999:
                    changing_factor = factor

                    for k, m in enumerate(metrics):
                        disturbed_ranking = create_mcda_ranking(mcda, m, NormalizedWeights(disturbed_weights), is_cost)
                        changed_ranking.append(disturbed_ranking.tolist())

                        if rankings_for_borda is not None and len(rankings_for_borda) > 0:
                            borda_rank = borda_count_rank([disturbed_ranking] + rankings_for_borda[k], borda_count_weights)
                            changed_borda_ranking.append(borda_rank.tolist())

                    break

                factor *= factor_adjustment

            changing_factors.append(changing_factor)
            changed_rankings.append(changed_ranking)
            changed_borda_rankings.append(changed_borda_ranking)

        return changing_factors, changed_rankings, changed_borda_rankings

    decreasing_iterations = math.floor(math.log10(lower_bound) / math.log(1.0 - step_size))
    increasing_iterations = math.floor(math.log10(upper_bound) / math.log(1.0 + step_size))

    decreasing_factors, decreasing_ranks, decreasing_borda_ranks = find_factors(1.0 - step_size, decreasing_iterations)
    increasing_factors, increasing_ranks, increasing_borda_ranks = find_factors(1.0 + step_size, increasing_iterations)

    return decreasing_factors, decreasing_ranks, decreasing_borda_ranks, increasing_factors, increasing_ranks, increasing_borda_ranks
