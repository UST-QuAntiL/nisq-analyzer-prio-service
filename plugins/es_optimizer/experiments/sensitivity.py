import json
import os
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
from pymcdm.methods import TOPSIS, PROMETHEE_II
from pymcdm.methods.mcda_method import MCDA_method

from plugins.es_optimizer import rank_correlation
from plugins.es_optimizer.experiments.tools.data_loader import load_csv_and_add_headers, \
    get_metrics_and_histogram_intersections, is_cost
from plugins.es_optimizer.experiments.tools.ranking import create_mcda_ranking, convert_scores_to_ranking
from plugins.es_optimizer.weights import NormalizedWeights


def calculate_average_spearman(mcda: MCDA_method, metrics: List[np.ndarray], original_weights: NormalizedWeights, disturbed_weights: NormalizedWeights) -> float:
    average_spearman = 0

    for m in metrics:
        original_ranking = create_mcda_ranking(mcda, m, original_weights, is_cost)
        disturbed_ranking = create_mcda_ranking(mcda, m, disturbed_weights, is_cost)

        spearman = rank_correlation.spearman(original_ranking, disturbed_ranking)
        average_spearman += spearman

    average_spearman /= len(metrics)

    return average_spearman


def find_changing_factors(mcda: MCDA_method, metrics: List[np.ndarray], original_weights: NormalizedWeights) -> Tuple[List[float], List[float]]:
    def find_factors(factor_adjustment: float) -> List[float]:
        changing_factors = []

        for i in range(original_weights.normalized_weights.shape[0]):
            factor = 1.0
            changing_factor = float("nan")

            for j in range(500):
                disturbed_weights: np.ndarray = original_weights.normalized_weights.copy()
                disturbed_weights[i] *= factor
                disturbed_weights /= np.sum(disturbed_weights)

                avg_spearman = calculate_average_spearman(
                    mcda, metrics, original_weights, NormalizedWeights(disturbed_weights))

                if avg_spearman < 0.99999:
                    changing_factor = factor
                    break

                factor *= factor_adjustment

            changing_factors.append(changing_factor)

        return changing_factors

    decreasing_factors = find_factors(0.99)
    increasing_factors = find_factors(1.01)

    print(decreasing_factors)
    print(increasing_factors)

    return decreasing_factors, increasing_factors


def main(dataset_path: str, learned_weights_path: str):
    data = load_csv_and_add_headers(dataset_path)
    metrics, histogram_intersections = get_metrics_and_histogram_intersections(data)
    mcda = None
    learned_weights_file_name = os.path.basename(learned_weights_path)

    if "topsis" in learned_weights_file_name.lower():
        mcda = TOPSIS()
    elif "promethee" in learned_weights_file_name.lower():
        mcda = PROMETHEE_II("usual")

    learned_weights_result = json.load(open(learned_weights_path))
    original_weights = [NormalizedWeights(np.array(weights)) for weights in learned_weights_result["normalized_weights"]]

    with Pool(8) as p:
        changing_factors_decrease, changing_factors_increase = zip(*p.starmap(
            find_changing_factors,
            zip(
                [mcda] * len(original_weights),
                [metrics] * len(original_weights),
                original_weights
            )
        ))

    json.dump({
        "changing_factors_decrease": changing_factors_decrease,
        "changing_factors_decrease_mean": np.nanmean(changing_factors_decrease, axis=0).tolist(),
        "changing_factors_decrease_std": np.nanstd(changing_factors_decrease, axis=0).tolist(),
        "changing_factors_decrease_se": (np.nanstd(changing_factors_decrease, axis=0) / np.sqrt(len(changing_factors_decrease))).tolist(),
        "changing_factors_decrease_nan_ratio": np.mean(np.isnan(changing_factors_decrease), axis=0).tolist(),

        "changing_factors_increase": changing_factors_increase,
        "changing_factors_increase_mean": np.nanmean(changing_factors_increase, axis=0).tolist(),
        "changing_factors_increase_std": np.nanstd(changing_factors_increase, axis=0).tolist(),
        "changing_factors_increase_se": (np.nanstd(changing_factors_increase, axis=0) / np.sqrt(len(changing_factors_increase))).tolist(),
        "changing_factors_increase_nan_ratio": np.mean(np.isnan(changing_factors_increase), axis=0).tolist(),
    }, open(os.path.join(os.path.dirname(learned_weights_path), "sensitivity_" + learned_weights_file_name), mode="wt"))


if __name__ == "__main__":
    main("data/Result_25.csv", "results/Result_25/TOPSIS_es.json")
