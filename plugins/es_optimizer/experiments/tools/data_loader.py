import math
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame

all_column_names = [
        "id1", "id2", "circuit_name", "qpu", "compiler", "result", "shots", "histogram_intersection", "width", "depth",
        "multi_qubit_gate_depth", "total_number_of_operations", "number_of_single_qubit_gates",
        "number_of_multi_qubit_gates", "number_of_measurement_operations", "single_qubit_gate_error",
        "multi_qubit_gate_error", "single_qubit_gate_time", "multi_qubit_gate_time", "readout_error", "t1", "t2",
        "queue_size"]
metric_column_names = all_column_names[8:22]
is_cost = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0])
dtypes = {
    "id1": object,
    "id2": object,
    "circuit_name": object,
    "qpu": object,
    "compiler": object,
    "result": object,
    "shots": float,
    "histogram_intersection": float,
    "width": float,
    "depth": float,
    "multi_qubit_gate_depth": float,
    "total_number_of_operations": float,
    "number_of_single_qubit_gates": float,
    "number_of_multi_qubit_gates": float,
    "number_of_measurement_operations": float,
    "single_qubit_gate_error": float,
    "multi_qubit_gate_error": float,
    "single_qubit_gate_time": float,
    "multi_qubit_gate_time": float,
    "readout_error": float,
    "t1": float,
    "t2": float,
    "queue_size": float
}


def load_csv_and_add_headers(file_path: str) -> DataFrame:
    data = pd.read_csv(file_path, header=None, names=all_column_names, dtype=dtypes)

    return data


def get_circuit_names(data: DataFrame) -> List[str]:
    names = list(set(data["circuit_name"]))

    return names


def _filter_out_simulators(data: DataFrame) -> DataFrame:
    return data[data["qpu"] != "ibmq_qasm_simulator"]


def _filter_out_compilations_with_missing_data(data: DataFrame) -> DataFrame:
    data = data[data["result"].notna()]
    data = data[data["histogram_intersection"] > 0]
    data = data[data["shots"] > 0]

    return data


# TODO: add training / test set split
# TODO: add cross validation
def get_metrics_and_histogram_intersections(data: DataFrame) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    data = _filter_out_compilations_with_missing_data(data)

    circuit_names = get_circuit_names(data)
    metrics = []
    histogram_intersections = []

    for circuit_name in circuit_names:
        single_circuit = data[data["circuit_name"] == circuit_name]
        single_circuit_metrics = single_circuit[metric_column_names]
        metrics.append(single_circuit_metrics.to_numpy(dtype=float))
        histogram_intersections.append(single_circuit["histogram_intersection"].to_numpy(dtype=float))

    return metrics, histogram_intersections


def convert_weights_array_to_dict(weights: np.ndarray) -> Dict[str, float]:
    weights_dict = {}

    for metric_name, weight in zip(metric_column_names, weights):
        weights_dict[metric_name] = weight

    return weights_dict


def create_random_training_test_split(metrics: List[np.ndarray], histogram_intersections: List[np.ndarray], training_ratio: float) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    assert len(metrics) == len(histogram_intersections)
    dataset_size = len(metrics)
    training_set_size = math.floor(dataset_size * training_ratio)

    rng = np.random.default_rng()
    indices = np.arange(dataset_size)
    rng.shuffle(indices)

    training_metrics = []
    training_histogram_intersections = []
    test_metrics = []
    test_histogram_intersections = []

    for i in indices[0:training_set_size]:
        training_metrics.append(metrics[i])
        training_histogram_intersections.append(histogram_intersections[i])

    for i in indices[training_set_size:]:
        test_metrics.append(metrics[i])
        test_histogram_intersections.append(histogram_intersections[i])

    return training_metrics, training_histogram_intersections, test_metrics, test_histogram_intersections


def main():
    data = load_csv_and_add_headers("Result_old.csv")
    data = _filter_out_simulators(data)
    metrics, histogram_intersections = get_metrics_and_histogram_intersections(data)
    create_random_training_test_split(metrics, histogram_intersections, 0.7)


if __name__ == "__main__":
    main()
