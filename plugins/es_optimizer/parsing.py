from typing import List, Dict, Tuple

import numpy as np


def get_metrics_from_compiled_circuits(compiled_circuits: List[Dict], metric_names: List[str]) -> np.ndarray:
    metrics = np.zeros((len(compiled_circuits), len(metric_names)), dtype=float)

    for i, compiled_circuit in enumerate(compiled_circuits):
        for j, metric_name in enumerate(metric_names):
            metrics[i, j] = compiled_circuit[metric_name]

    return metrics


def get_histogram_intersections_from_compiled_circuits(compiled_circuits: List[Dict]) -> np.ndarray:
    histogram_intersections_list = [circ["histogramIntersection"] for circ in compiled_circuits]

    return np.array(histogram_intersections_list, dtype=float)


def parse_metric_info(task_parameters: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    weights = np.zeros(len(task_parameters["metrics"]), dtype=float)
    is_cost = np.zeros(len(task_parameters["metrics"]), dtype=float)  # 1.0 = profit, -1.0 = cost
    metric_names = []
    metric_index = 0

    for metric_name, metric_data in task_parameters["metrics"].items():
        weights[metric_index] = metric_data["weight"]
        is_cost[metric_index] = -1.0 if metric_data["is_cost"] is True else 1.0
        metric_names.append(metric_name)

        metric_index += 1

    return weights, is_cost, metric_names
