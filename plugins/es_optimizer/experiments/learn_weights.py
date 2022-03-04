from pprint import pprint
from typing import List

import numpy as np
from pymcdm.methods import TOPSIS, PROMETHEE_II
from pymcdm.methods.mcda_method import MCDA_method
from scipy.optimize import minimize
from sklearn import preprocessing

from plugins.es_optimizer.evolutionary_strategy import evolutionary_strategy
from plugins.es_optimizer.experiments.data_loader import load_csv_and_add_headers, \
    get_metrics_and_histogram_intersections, convert_weights_array_to_dict, is_cost
from plugins.es_optimizer.objective_functions import objective_function_all_circuits
from plugins.es_optimizer.standard_genetic_algorithm import standard_genetic_algorithm

mcda_methods = [TOPSIS(), PROMETHEE_II("usual")]
learning_methods = ["es", "ga", "COBYLA"]


def learn_best_weights(
    learning_method: str, mcda: MCDA_method, metrics: List[np.ndarray],
    histogram_intersections: List[np.ndarray], is_cost: np.ndarray) -> np.ndarray:
    metrics_cnt = metrics[0].shape[1]

    if learning_method == "es":
        best_weights = evolutionary_strategy(mcda, metrics, histogram_intersections, is_cost)
    elif learning_method == "ga":
        best_weights = standard_genetic_algorithm(mcda, metrics, histogram_intersections, is_cost)
    else:
        result = minimize(
            objective_function_all_circuits, np.random.random(metrics_cnt),
            (mcda, metrics, histogram_intersections, is_cost), method=learning_method,
            options={"disp": True})
        best_weights = preprocessing.MinMaxScaler().fit_transform(result.x.reshape((-1, 1))).reshape((-1))
        best_weights /= np.sum(best_weights)

    return best_weights


def main():
    data = load_csv_and_add_headers("Result_old.csv")
    metrics, histogram_intersections = get_metrics_and_histogram_intersections(data)

    weights = learn_best_weights(learning_methods[0], mcda_methods[0], metrics, histogram_intersections, is_cost)
    weights_dict = convert_weights_array_to_dict(weights)

    pprint(weights_dict)


if __name__ == "__main__":
    main()
