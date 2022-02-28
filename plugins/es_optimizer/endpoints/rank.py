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
from pymcdm.methods import TOPSIS, PROMETHEE_II
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_result, save_task_error

from plugins.es_optimizer.api import PLUGIN_BLP, RankSchema
from plugins.es_optimizer.parsing import get_metrics_from_compiled_circuits, parse_metric_info
from plugins.es_optimizer.plugin import EsOptimizer


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


TASK_LOGGER = get_task_logger(__name__)


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
        scores = TOPSIS()(metrics, weights, is_cost)
    elif task_parameters["method"] == "promethee_ii":
        scores = PROMETHEE_II("usual")(metrics, weights, is_cost)
    else:
        raise ValueError("Unknown method: " + str(task_parameters["method"]))

    output_data = {}
    compiled_circuit_ids = [circ["id"] for circ in compiled_circuits]

    for compiled_circuit_id, score in zip(compiled_circuit_ids, scores):
        output_data[compiled_circuit_id] = score

    with SpooledTemporaryFile(mode="wt") as output_file:
        json.dump(output_data, output_file)
        STORE.persist_task_result(
            db_id, output_file, "scores.json", "text", "application/json"
        )

    return "finished"
