# when renaming this module also change the import in __init__.py
import json
from http import HTTPStatus
from json import dumps, loads
from tempfile import SpooledTemporaryFile
from datetime import datetime
from typing import Any, Mapping, Optional, List, Dict, Tuple

import numpy as np
from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import redirect
from flask.helpers import url_for
from flask.views import MethodView
from marshmallow import EXCLUDE
from pymcdm.methods import TOPSIS

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result

################################################################################
# import plugin specific requirements in this module
# the get_requirements method of the plugin class must be defined and functional
# before any of the requirements specified in the method are imported!
################################################################################

from .api import PLUGIN_BLP, RankSchema
from .plugin import EsOptimizer


@PLUGIN_BLP.route("/rank")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @PLUGIN_BLP.arguments(RankSchema(unknown=EXCLUDE), location="json")
    @PLUGIN_BLP.response(HTTPStatus.SEE_OTHER)
    @PLUGIN_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the background task."""
        # create a new task instance in DB with the relevant parameters
        db_task = ProcessingTask(task_name=rank_task.name, parameters=dumps(arguments))
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = rank_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors appearing somewhere in task chain to db
        task.link_error(save_task_error.s(db_id=db_task.id))

        try:
            # start the task chain as a background task
            task.apply_async()
        except Exception as e:
            # save error in DB if task could not be scheduled!
            db_task.task_status = "FAILURE"
            db_task.finished_at = datetime.utcnow()
            db_task.add_task_log_entry(f"Error scheduling task: {e!r}")

            db_task.save(commit=True)
            raise e  # and raise exception again

        # redirect to the created task resource (constructed from the ProcessingTask saved in the DB)
        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@PLUGIN_BLP.route("/learn-ranking")
class LearnRankingView(MethodView):
    """Start a long running processing task."""

    @PLUGIN_BLP.arguments(RankSchema(unknown=EXCLUDE), location="json")
    @PLUGIN_BLP.response(HTTPStatus.SEE_OTHER)
    @PLUGIN_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the background task."""
        # create a new task instance in DB with the relevant parameters
        db_task = ProcessingTask(task_name=learn_ranking_task.name, parameters=dumps(arguments))
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = learn_ranking_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors appearing somewhere in task chain to db
        task.link_error(save_task_error.s(db_id=db_task.id))

        try:
            # start the task chain as a background task
            task.apply_async()
        except Exception as e:
            # save error in DB if task could not be scheduled!
            db_task.task_status = "FAILURE"
            db_task.finished_at = datetime.utcnow()
            db_task.add_task_log_entry(f"Error scheduling task: {e!r}")

            db_task.save(commit=True)
            raise e  # and raise exception again

        # redirect to the created task resource (constructed from the ProcessingTask saved in the DB)
        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


TASK_LOGGER = get_task_logger(__name__)


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


# task names must be globally unique => use full versioned plugin identifier to scope name
@CELERY.task(name=f"{EsOptimizer.instance.identifier}.rank_task", bind=True)
def rank_task(self, db_id: int) -> str:
    """The main background task of the plugin ES Optimizer."""
    TASK_LOGGER.info(f"Starting new background task for plugin ES Optimizer with db id '{db_id}'")

    # load task data based on given DB id
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    # deserialize task parameters
    task_parameters: Dict[str, Any] = loads(task_data.parameters or "{}")

    weights, is_cost, metric_names = parse_metric_info(task_parameters)

    compiled_circuits = task_parameters["circuits"][0]["compiled_circuits"]
    metrics = get_metrics_from_compiled_circuits(compiled_circuits, metric_names)

    if task_parameters["method"] == "topsis":
        topsis = TOPSIS()
        scores = topsis(metrics, weights, is_cost)
        output_data = {}
        compiled_circuit_ids = [circ["id"] for circ in compiled_circuits]

        for compiled_circuit_id, score in zip(compiled_circuit_ids, scores):
            output_data[compiled_circuit_id] = score

        with SpooledTemporaryFile(mode="wt") as output_file:
            json.dump(output_data, output_file)
            STORE.persist_task_result(
                db_id, output_file, "scores.json", "text", "application/json"
            )
    else:
        raise ValueError("Unknown method: " + str(task_parameters["method"]))

    return "finished"


def objective_function(metrics: np.ndarray, histogram_intersection: np.ndarray, weights: np.ndarray, is_cost: np.ndarray) -> float:
    target_scores = histogram_intersection
    topsis_scores = TOPSIS()(metrics, weights, is_cost)

    # normalization
    target_scores -= np.min(target_scores)
    target_scores /= np.max(target_scores)

    topsis_scores -= np.min(topsis_scores)
    topsis_scores /= np.max(topsis_scores)

    # mean square error
    loss = np.mean((target_scores - topsis_scores) * (target_scores - topsis_scores))

    return loss.item()


def objective_function_array(
    metrics: np.ndarray, histogram_intersection: np.ndarray, weights: np.ndarray, is_cost: np.ndarray) -> np.ndarray:
    return np.array([objective_function(metrics, histogram_intersection, w, is_cost) for w in weights], dtype=float)


def objective_function_all_circuits(
    weights: np.ndarray, metrics: List[np.ndarray], histogram_intersections: List[np.ndarray], is_cost: np.ndarray) -> float:
    error = 0.0

    for i in range(len(metrics)):
        error += objective_function(metrics[i], histogram_intersections[i], weights, is_cost)

    error = error / len(metrics)

    return error


def evolutionary_strategy(metrics: List[np.ndarray], histogram_intersection: List[np.ndarray], is_cost: np.ndarray) -> np.ndarray:
    population_size = 20
    reproduction_factor = 4
    mutation_factor = 0.05
    metrics_cnt = metrics[0].shape[1]
    weights = np.random.random((population_size, metrics_cnt))

    for i in range(100):
        obj_values = objective_function_array(metrics[0], histogram_intersection[0], weights, is_cost)

        for m, hi in zip(metrics[1:], histogram_intersection[1:]):
            obj_values += objective_function_array(m, hi, weights, is_cost)

        obj_values /= len(metrics)

        sorted_indices = np.argsort(obj_values)

        obj_values = obj_values[sorted_indices]
        weights = weights[sorted_indices]

        # remove worst weights
        weights: np.ndarray = weights[0:population_size // reproduction_factor]

        # clone and mutate weights
        new_weights = [weights]

        for i in range(reproduction_factor - 1):
            new_weights.append(weights.copy() + np.random.normal(scale=mutation_factor, size=weights.shape))

        weights = np.concatenate(new_weights, axis=0)

    return weights[0]


@CELERY.task(name=f"{EsOptimizer.instance.identifier}.learn_ranking_task", bind=True)
def learn_ranking_task(self, db_id: int) -> str:
    """The main background task of the plugin ES Optimizer."""
    TASK_LOGGER.info(f"Starting new background task for plugin ES Optimizer with db id '{db_id}'")

    # load task data based on given DB id
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    # deserialize task parameters
    task_parameters: Dict[str, Any] = loads(task_data.parameters or "{}")

    weights, is_cost, metric_names = parse_metric_info(task_parameters)
    metrics = []
    histogram_intersections = []

    for circuit in task_parameters["circuits"]:
        compiled_circuits = circuit["compiled_circuits"]
        metrics.append(get_metrics_from_compiled_circuits(compiled_circuits, metric_names))
        histogram_intersections.append(get_histogram_intersections_from_compiled_circuits(compiled_circuits))

    if task_parameters["method"] == "topsis":
        best_weights = evolutionary_strategy(metrics, histogram_intersections, is_cost)

        with SpooledTemporaryFile(mode="wt") as output_file:
            metric_weights = {}

            for name, weight in zip(metric_names, best_weights):
                metric_weights[name] = weight

            json.dump(metric_weights, output_file)
            STORE.persist_task_result(
                db_id, output_file, "weights.json", "text", "application/json"
            )
    else:
        raise ValueError("Unknown method: " + str(task_parameters["method"]))

    return "finished"
