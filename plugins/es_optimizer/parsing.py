from typing import List, Dict, Tuple

import numpy as np

from .tools.ranking import convert_scores_to_ranking


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
    weights = np.zeros(len(task_parameters["metric_weights"]), dtype=float)
    is_cost = np.zeros(len(task_parameters["metric_weights"]), dtype=float)  # 1.0 = profit, -1.0 = cost
    metric_names = []
    metric_index = 0

    for metric_name, metric_data in task_parameters["metric_weights"].items():
        weights[metric_index] = metric_data["weight"]
        is_cost[metric_index] = -1.0 if metric_data["is_cost"] is True else 1.0
        metric_names.append(metric_name)

        metric_index += 1

    return weights, is_cost, metric_names


def get_rankings_for_borda_count(task_parameters: Dict, circuit_index: int) -> List[np.ndarray]:
    circuit_data = task_parameters["circuits"][circuit_index]
    borda_metric_names: List[str] = list(task_parameters["borda_count_metrics"].keys())
    borda_metrics = get_metrics_from_compiled_circuits(circuit_data["compiled_circuits"], borda_metric_names)
    rankings = []

    for i, name in enumerate(borda_metric_names):
        is_cost = task_parameters["borda_count_metrics"][name]["is_cost"]
        rankings.append(convert_scores_to_ranking(borda_metrics[:, i].reshape((-1)), not is_cost))

    return rankings


if __name__ == "__main__":
    task_parameters = {
        "circuits": [
            {
                "id": "shor-fix-15",
                "compiled_circuits": [
                    {
                        "id": "ibmq_manila:pytket",
                        "histogramIntersection": 0.967407227,
                        "width": 5,
                        "depth": 11,
                        "multiqdepth": 2,
                        "total num ops": 13,
                        "# single g": 8,
                        "# multi g": 2,
                        "# meas ops": 3,
                        "single err": 0.000191173,
                        "multi err": 0.007548308,
                        "single g time": 26.66666667,
                        "mulit g time": 350.2222222,
                        "readout err": 0.02762,
                        "t1": 195.8578339,
                        "t2": 59.11248848
                    },
                    {
                        "id": "ibmq_lima:pytket",
                        "histogramIntersection": 0.966674805,
                        "width": 5,
                        "depth": 11,
                        "multiqdepth": 2,
                        "total num ops": 13,
                        "# single g": 8,
                        "# multi g": 2,
                        "# meas ops": 3,
                        "single err": 0.000255855,
                        "multi err": 0.010499085,
                        "single g time": 26.66666667,
                        "mulit g time": 387.5555556,
                        "readout err": 0.02792,
                        "t1": 93.93756879,
                        "t2": 103.4710873
                    },
                    {
                        "id": "ibmq_quito:pytket",
                        "histogramIntersection": 0.962524414,
                        "width": 5,
                        "depth": 11,
                        "multiqdepth": 2,
                        "total num ops": 13,
                        "# single g": 8,
                        "# multi g": 2,
                        "# meas ops": 3,
                        "single err": 0.00083408,
                        "multi err": 0.024837885,
                        "single g time": 26.66666667,
                        "mulit g time": 277.3333333,
                        "readout err": 0.05634,
                        "t1": 100.7355939,
                        "t2": 106.766057
                    },
                    {
                        "id": "ibmq_qasm_simulator:qiskit",
                        "histogramIntersection": 1,
                        "width": 5,
                        "depth": 7,
                        "multiqdepth": 4,
                        "total num ops": 10,
                        "# single g": 2,
                        "# multi g": 5,
                        "# meas ops": 3,
                        "single err": 0,
                        "multi err": 0,
                        "single g time": 0,
                        "mulit g time": 0,
                        "readout err": 0,
                        "t1": 99999,
                        "t2": 99999
                    },
                    {
                        "id": "ibmq_manila:qiskit",
                        "histogramIntersection": 0.956787109,
                        "width": 4,
                        "depth": 23,
                        "multiqdepth": 9,
                        "total num ops": 36,
                        "# single g": 24,
                        "# multi g": 9,
                        "# meas ops": 3,
                        "single err": 0.000191173,
                        "multi err": 0.007548308,
                        "single g time": 26.66666667,
                        "mulit g time": 350.2222222,
                        "readout err": 0.02762,
                        "t1": 195.8578339,
                        "t2": 59.11248848
                    },
                    {
                        "id": "ibmq_lima:qiskit",
                        "histogramIntersection": 0.920776367,
                        "width": 5,
                        "depth": 16,
                        "multiqdepth": 6,
                        "total num ops": 27,
                        "# single g": 15,
                        "# multi g": 9,
                        "# meas ops": 3,
                        "single err": 0.000255855,
                        "multi err": 0.010499085,
                        "single g time": 26.66666667,
                        "mulit g time": 387.5555556,
                        "readout err": 0.02792,
                        "t1": 93.93756879,
                        "t2": 103.4710873
                    },
                    {
                        "id": "ibmq_quito:qiskit",
                        "histogramIntersection": 0.852294922,
                        "width": 5,
                        "depth": 20,
                        "multiqdepth": 11,
                        "total num ops": 30,
                        "# single g": 13,
                        "# multi g": 14,
                        "# meas ops": 3,
                        "single err": 0.00083408,
                        "multi err": 0.024837885,
                        "single g time": 26.66666667,
                        "mulit g time": 277.3333333,
                        "readout err": 0.05634,
                        "t1": 100.7355939,
                        "t2": 106.766057
                    }
                ]
            }
        ],
        "borda_count_metrics": {
            "total num ops": {
                "is_cost": True
            },
            "multiqdepth": {
                "is_cost": True
            }
        }
    }

    test = get_rankings_for_borda_count(task_parameters, 0)
