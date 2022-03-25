import json

import numpy as np
from pandas import DataFrame
from pymcdm.methods import TOPSIS
from pymcdm.methods.mcda_method import MCDA_method

from ..tools.ranking import convert_scores_to_ranking, create_mcda_ranking
from ..borda_count import borda_count_rank
from .tools.data_loader import load_csv_and_add_headers, is_cost
from ..weights import NormalizedWeights


class SingleCircuitData:
    def __init__(self, data: DataFrame, circ_name: str):
        data = data[data["circuit_name"] == circ_name]
        data = data[data["qpu"] != "ibmq_qasm_simulator"]
        data = data[data["shots"] > 0]

        metric_columns = data.columns[8:22]
        self.metrics = data[metric_columns]
        self.histogram_intersections = data["histogram_intersection"]
        self.queue_sizes = data["queue_size"]

    def rank_histogram_intersections(self) -> np.ndarray:
        return convert_scores_to_ranking(self.histogram_intersections.to_numpy(), higher_scores_are_better=True)

    def rank_queue_sizes(self) -> np.ndarray:
        return convert_scores_to_ranking(self.queue_sizes.to_numpy(), higher_scores_are_better=False)

    def rank_with_mcda(self, mcda: MCDA_method, weights: NormalizedWeights, is_cost: np.ndarray) -> np.ndarray:
        return create_mcda_ranking(mcda, self.metrics.to_numpy(), weights, is_cost)


def load_weights_mean(file_path: str) -> np.ndarray:
    results = json.load(open(file_path, mode="rt"))

    return np.array(results["weights_mean"])


def main():
    data = load_csv_and_add_headers("data/Result_25.csv")
    circ_name = "shor-fix-15-qiskit"
    sc_data = SingleCircuitData(data, circ_name)
    weights_mean = load_weights_mean("results/Result_25/TOPSIS_COBYLA.json")

    histo_rank = sc_data.rank_histogram_intersections()
    mcda_rank = sc_data.rank_with_mcda(TOPSIS(), NormalizedWeights(weights_mean), is_cost)
    queue_rank = sc_data.rank_queue_sizes()
    borda_rank = borda_count_rank([mcda_rank, queue_rank])

    print("histo inter rank:", histo_rank)
    print("MCDA rank:", mcda_rank)
    print("queue rank:", queue_rank)
    print("borda rank:", borda_rank)


if __name__ == "__main__":
    main()
