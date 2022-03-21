import json
import os
from multiprocessing import Pool

import numpy as np
from pymcdm.methods import TOPSIS, PROMETHEE_II

from plugins.es_optimizer.experiments.tools.data_loader import load_csv_and_add_headers, \
    get_metrics_and_histogram_intersections
from plugins.es_optimizer.sensitivity import find_changing_factors
from plugins.es_optimizer.weights import NormalizedWeights


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
        changing_factors_decrease, _, changing_factors_increase, _ = zip(*p.starmap(
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
