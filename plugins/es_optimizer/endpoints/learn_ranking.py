import json
from datetime import datetime
from http import HTTPStatus
from json import dumps, loads
from tempfile import SpooledTemporaryFile
from typing import Optional, Dict, Any

from celery import chain
from celery.utils.log import get_task_logger
from flask import redirect, url_for
from flask.views import MethodView
from marshmallow import EXCLUDE
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_result, save_task_error

from ..api import PLUGIN_BLP
from ..schemas import LearnRankingSchema
from ..plugin import EsOptimizer


@PLUGIN_BLP.route("/learn-ranking")
class LearnRankingView(MethodView):
    """
    Description
    -----------
    This endpoint will optimize weights for a MCDA method for compiled quantum circuits. The optimization goal is that
    the MCDA method gives the same (normalized) scores as the given histogram intersections.

    Input
    -----
    mcda_method: MCDA method for which weights will be optimized
    learning_method: learning method to be used to optimize the weights (es, ga, scipy minimizer)
    metric_weights: defines the metrics to be used and whether they represent a cost or not
        [metric name]: (histogram_intersection must be present)
            weight: this value will be ignored
            is_cost: true = lower is better, false = higher is better
    circuits: multiple circuit implementations and their compiled circuits and QPU metrics

    Output
    ------
    [metric name]:
        normalized_weight: learned weights for the MCDA method
        is_cost: true = lower is better, false = higher is better
    """

    @PLUGIN_BLP.arguments(LearnRankingSchema(unknown=EXCLUDE), location="json")
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
        response = redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )

        response.autocorrect_location_header = True

        return response


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{EsOptimizer.instance.identifier}.learn_ranking_task", bind=True)
def learn_ranking_task(self, db_id: int) -> str:
    import numpy as np
    from pymcdm.methods import TOPSIS, PROMETHEE_II
    from scipy.optimize import minimize
    from ..evolutionary_strategy import evolutionary_strategy
    from ..objective_functions import objective_function_all_circuits
    from ..parsing import get_metrics_from_compiled_circuits, get_histogram_intersections_from_compiled_circuits, \
        parse_metric_info
    from ..standard_genetic_algorithm import standard_genetic_algorithm
    from ..weights import Weights

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

    if task_parameters["mcda_method"] == "topsis":
        mcda = TOPSIS()
    elif task_parameters["mcda_method"] == "promethee_ii":
        mcda = PROMETHEE_II("usual")
    else:
        msg = "Unknown MCDA method: " + str(task_parameters["mcda_method"])
        TASK_LOGGER.error(msg)
        raise ValueError(msg)

    if task_parameters["learning_method"] == "es":
        best_weights = evolutionary_strategy(mcda, metrics, histogram_intersections, is_cost)
    elif task_parameters["learning_method"] == "ga":
        best_weights = standard_genetic_algorithm(mcda, metrics, histogram_intersections, is_cost)
    else:
        result = minimize(
            objective_function_all_circuits, np.random.random(weights.shape), (mcda, metrics, histogram_intersections, is_cost), method=task_parameters["learning_method"],
            options={"disp": True})
        best_weights = Weights.normalize(result.x)

    with SpooledTemporaryFile(mode="wt") as output_file:
        metric_weights = {}

        for name, ic, weight in zip(metric_names, is_cost, best_weights.normalized_weights):
            metric_weights[name] = {
                "normalized_weight": weight,
                "isCost": True if ic == -1.0 else False
            }

        json.dump(metric_weights, output_file)
        STORE.persist_task_result(
            db_id, output_file, "weights.json", "text", "application/json"
        )

    return "finished"
