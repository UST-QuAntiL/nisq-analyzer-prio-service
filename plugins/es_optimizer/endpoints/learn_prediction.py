import json
from datetime import datetime
from http import HTTPStatus
from tempfile import SpooledTemporaryFile
from typing import Optional

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
from ..schemas import LearnPredictionSchema, LearnPrediction, MachineLearningMethod
from ..plugin import EsOptimizer


@PLUGIN_BLP.route("/prediction")
class PredictionView(MethodView):
    """Start a long running processing task."""

    @PLUGIN_BLP.arguments(LearnPredictionSchema(unknown=EXCLUDE), location="json")
    @PLUGIN_BLP.response(HTTPStatus.SEE_OTHER)
    @PLUGIN_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the background task."""
        # create a new task instance in DB with the relevant parameters
        schema = LearnPredictionSchema()
        db_task = ProcessingTask(task_name=prediction_task.name, parameters=schema.dumps(arguments))
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = prediction_task.s(db_id=db_task.id) | save_task_result.s(
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


def _preprocess_data(task_parameters: LearnPrediction):
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np

    training_input_dict = {}

    for name in task_parameters.input_metric_names:
        metric_data = []

        for circuits in task_parameters.training_data:
            for circuit in circuits.original_circuit_and_qpu_metrics:
                metric_data.append(circuit[name])

        training_input_dict[name] = metric_data

    compiler_data = []

    for circuits in task_parameters.training_data:
        for circuit in circuits.original_circuit_and_qpu_metrics:
            compiler_data.append(circuit[task_parameters.compiler_property_name])

    compiler_encoder = OneHotEncoder(sparse=False)
    compilers_encoded: np.ndarray = compiler_encoder.fit_transform(np.array(compiler_data).reshape(-1, 1))

    for i in range(compilers_encoded.shape[1]):
        training_input_dict[f"compiler_{i}"] = compilers_encoded[:, i]

    training_input = pd.DataFrame(training_input_dict)

    target_values = []

    for circuits in task_parameters.training_data:
        for circuit in circuits.original_circuit_and_qpu_metrics:
            target_values.append(circuit[task_parameters.histogram_intersection_name])

    training_target_dict = {
        task_parameters.histogram_intersection_name: target_values
    }

    training_target = pd.DataFrame(training_target_dict)

    new_circuit_data = task_parameters.new_circuit.original_circuit_and_qpu_metrics[0]
    new_input = pd.DataFrame(new_circuit_data)

    return training_input, training_target, new_input, compiler_encoder


@CELERY.task(name=f"{EsOptimizer.instance.identifier}.prediction_task", bind=True)
def prediction_task(self, db_id: int) -> str:
    # import pydevd_pycharm
    # pydevd_pycharm.settrace('localhost', port=3857, stdoutToServer=True, stderrToServer=True)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor

    """The background task that trains a machine learning model to predict histogram intersections."""
    TASK_LOGGER.info(f"Starting new background task for plugin ES Optimizer with db id '{db_id}'")

    # load task data based on given DB id
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    # deserialize task parameters
    schema = LearnPredictionSchema()
    task_parameters: LearnPrediction = schema.loads(task_data.parameters)

    training_input, training_target, new_input, compiler_encoder = _preprocess_data(task_parameters)

    if task_parameters.machine_learning_method == MachineLearningMethod.extra_trees_regressor:
        model = make_pipeline(StandardScaler(), ExtraTreesRegressor())
    elif task_parameters.machine_learning_method == MachineLearningMethod.random_forest_regressor:
        model = make_pipeline(StandardScaler(), RandomForestRegressor())
    elif task_parameters.machine_learning_method == MachineLearningMethod.gradient_boosting_regressor:
        model = make_pipeline(StandardScaler(), GradientBoostingRegressor())
    else:
        raise NotImplementedError

    model.fit(training_input, training_target)

    with SpooledTemporaryFile(mode="wt") as output_file:
        predicted_histogram_intersections = {}

        # TODO: add predicted histogram intersections to dictionary

        json.dump(predicted_histogram_intersections, output_file)
        STORE.persist_task_result(
            db_id, output_file, "weights.json", "text", "application/json"
        )

    return "finished"
