import json
from multiprocessing import Pool
from pprint import pprint
from typing import List

import numpy as np
from pymcdm.methods import TOPSIS, PROMETHEE_II
from pymcdm.methods.mcda_method import MCDA_method
from scipy.optimize import minimize

from plugins.es_optimizer import rank_correlation
from plugins.es_optimizer.evolutionary_strategy import evolutionary_strategy
from plugins.es_optimizer.experiments.tools import data_loader
from plugins.es_optimizer.experiments.tools.data_loader import load_csv_and_add_headers, \
    get_metrics_and_histogram_intersections, convert_weights_array_to_dict, is_cost, create_random_training_test_split
from plugins.es_optimizer.experiments.tools.ranking import create_mcda_ranking, convert_scores_to_ranking
from plugins.es_optimizer.objective_functions import objective_function_all_circuits
from plugins.es_optimizer.preprocessing import normalize_weights
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
        best_weights = normalize_weights(result.x)

    return best_weights


def calculate_average_spearman(mcda: MCDA_method, metrics: List[np.ndarray], histogram_intersections: List[np.ndarray], weights: np.ndarray) -> float:
    average_spearman = 0

    for m, hi in zip(metrics, histogram_intersections):
        mcda_ranking = create_mcda_ranking(mcda, m, weights, is_cost)
        target_ranking = convert_scores_to_ranking(hi, True)

        spearman = rank_correlation.spearman(mcda_ranking, target_ranking)
        average_spearman += spearman

    average_spearman /= len(metrics)

    return average_spearman


def calculate_standard_error(samples: List[float]) -> float:
    """
    :param samples: samples
    :return: standard error of the mean of the samples
    """
    return np.std(samples) / np.sqrt(len(samples))


def execute_training(metrics: List[np.ndarray], histogram_intersections: List[np.ndarray], mcda_method: MCDA_method, learning_method: str):
    training_metrics, training_histogram_intersections, test_metrics, test_histogram_intersections = \
        create_random_training_test_split(metrics, histogram_intersections, 0.7)
    weights = learn_best_weights(
        learning_method, mcda_method, training_metrics, training_histogram_intersections, is_cost)

    # weights_dict = convert_weights_array_to_dict(weights)
    # pprint(weights_dict)

    # print(create_mcda_ranking(mcda_methods[0], training_metrics[0], weights, is_cost))
    # print(convert_scores_to_ranking(training_histogram_intersections[0], True))

    new_training_spearman = \
        calculate_average_spearman(mcda_method, training_metrics, training_histogram_intersections, weights)
    new_test_spearman = \
        calculate_average_spearman(mcda_method, test_metrics, test_histogram_intersections, weights)
    print(new_training_spearman)
    print(new_test_spearman)
    print()

    return new_training_spearman, new_test_spearman, weights.tolist()


def main(mcda_method: MCDA_method, learning_method: str):
    data = load_csv_and_add_headers("data/Result_old.csv")
    metrics, histogram_intersections = get_metrics_and_histogram_intersections(data)
    iteration_cnt = 100

    with Pool(8) as p:
        training_spearman, test_spearman, weights_list = zip(*p.starmap(execute_training, [(metrics, histogram_intersections, mcda_method, learning_method)] * iteration_cnt))

    training_mean = np.mean(training_spearman)
    training_std = np.std(training_spearman)
    training_se = calculate_standard_error(training_spearman)
    test_mean = np.mean(test_spearman)
    test_std = np.std(test_spearman)
    test_se = calculate_standard_error(test_spearman)

    print("training mean: " + str(training_mean))
    print("training std : " + str(training_std))
    print("training standard error of estimated mean: " + str(training_se))
    print("test mean: " + str(test_mean))
    print("test std : " + str(test_std))
    print("test standard error of estimated mean: " + str(test_se))

    result = {
        "training_mean": training_mean,
        "training_std": training_std,
        "training_se": training_se,
        "test_mean": test_mean,
        "test_std": test_std,
        "test_se": test_se,
        "metric_names": data_loader.metric_column_names,
        "weights": weights_list
    }

    json.dump(result, open(mcda_method.__class__.__name__ + "_" + learning_method + ".json", mode="wt"))


def train_all_variations():
    for mcda_method in mcda_methods:
        for learning_method in learning_methods:
            main(mcda_method, learning_method)


if __name__ == "__main__":
    train_all_variations()
