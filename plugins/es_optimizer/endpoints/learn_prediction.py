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
from sklearn.ensemble import HistGradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import NuSVR
from sklearn.tree import DecisionTreeRegressor

from ..api import PLUGIN_BLP
from ..borda_count import borda_count_rank
from ..schemas import LearnPredictionSchema, LearnPrediction, MachineLearningMethod, PredictionResult, \
    PredictionResultSchema, MetaRegressor
from ..plugin import EsOptimizer
from ..tools.ranking import convert_scores_to_ranking, sort_array_with_ranking


@PLUGIN_BLP.route("/prediction")
class PredictionView(MethodView):
    """
    Description
    -----------
    Trains a regressor and optionally a meta-regressor on a dataset of multiple circuit implementations and compiled
    circuits to predict the histogram intersections for the compiled circuits of a new circuit implementation.

    Input
    -----
    machine_learning_method: regression ML method
    meta_regressor: meta regressor or none
    training_data: multiple implementations and there compiled circuits with QPU metrics to be used as training data
    new_circuit: new circuit implementation with its compiled circuits and QPU metrics for which the histogram
        intersections should be predicted
    input_metric_names: list of the metric names that will be used as input to the regressor
    compiler_property_name: key under which the compiler can be found
    histogram_intersection_name: key under which the histogram intersection can be found
    queue_size_name: key under which the queue size can be found
    queue_size_importance: factor which controls the influence of the queue size for the borda count ranking

    Output
    ------
    predicted_histogram_intersections: predicted histogram intersections for the new circuit implementation
    ranking: ranking based on the predicted histogram intersections
    borda_count_ranking: ranking that combines the predicted histogram intersections with the queue size

    """

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


def _convert_compilers_to_one_hot_encoding(data, task_parameters: LearnPrediction, encoder=None):
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np

    compiler_data = data[task_parameters.compiler_property_name].to_numpy().reshape((-1, 1))

    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False)
        compilers_encoded: np.ndarray = encoder.fit_transform(compiler_data)
    else:
        compilers_encoded: np.ndarray = encoder.transform(compiler_data)

    del data[task_parameters.compiler_property_name]
    column_names = []

    for i in range(compilers_encoded.shape[1]):
        data.insert(i, f"compiler_{i}", compilers_encoded[:, i])
        column_names.append(f"compiler_{i}")

    return encoder, column_names


def _preprocess_data(task_parameters: LearnPrediction):
    import pandas as pd

    circuit_data = []

    for circuits in task_parameters.training_data:
        for circuit in circuits.original_circuit_and_qpu_metrics:
            circuit_data.append(pd.DataFrame(circuit, index=[0]))

    training_data = pd.concat(circuit_data, axis=0, ignore_index=True)
    compiler_encoder, compiler_column_names = _convert_compilers_to_one_hot_encoding(training_data, task_parameters)

    training_input = training_data[task_parameters.input_metric_names + compiler_column_names]

    target_values = []

    for circuits in task_parameters.training_data:
        for circuit in circuits.original_circuit_and_qpu_metrics:
            target_values.append(circuit[task_parameters.histogram_intersection_name])

    training_target_dict = {
        task_parameters.histogram_intersection_name: target_values
    }

    training_target = pd.DataFrame(training_target_dict)

    new_circuit_data = []

    for circuit in task_parameters.new_circuit.original_circuit_and_qpu_metrics:
        new_circuit_data.append(pd.DataFrame(circuit, index=[0]))

    new_input = pd.concat(new_circuit_data, axis=0, ignore_index=True)

    new_ids = new_input["id"]
    new_queue_sizes = new_input[task_parameters.queue_size_name]

    _convert_compilers_to_one_hot_encoding(new_input, task_parameters, compiler_encoder)
    new_input = new_input[task_parameters.input_metric_names + compiler_column_names]

    return training_input, training_target, new_input, new_ids, new_queue_sizes, compiler_encoder


def _calculate_borda_rank(task_parameters: LearnPrediction, prediction_and_metadata):
    queue_rank = convert_scores_to_ranking(prediction_and_metadata["queue_size"].to_numpy(), False)
    prediction_rank = convert_scores_to_ranking(prediction_and_metadata["prediction"].to_numpy(), True)

    combined_rank = borda_count_rank(
        [prediction_rank, queue_rank],
        [1.0 - task_parameters.queue_size_importance, task_parameters.queue_size_importance])

    return combined_rank


@CELERY.task(name=f"{EsOptimizer.instance.identifier}.prediction_task", bind=True)
def prediction_task(self, db_id: int) -> str:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    import numpy as np
    import pandas as pd

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

    training_input, training_target, new_input, new_ids, new_queue_sizes, compiler_encoder = _preprocess_data(task_parameters)

    if task_parameters.machine_learning_method == MachineLearningMethod.extra_trees_regressor:
        regressor = ExtraTreesRegressor()
    elif task_parameters.machine_learning_method == MachineLearningMethod.random_forest_regressor:
        regressor = RandomForestRegressor()
    elif task_parameters.machine_learning_method == MachineLearningMethod.gradient_boosting_regressor:
        regressor = GradientBoostingRegressor()
    elif task_parameters.machine_learning_method == MachineLearningMethod.decision_tree_regressor:
        regressor = DecisionTreeRegressor()
    elif task_parameters.machine_learning_method == MachineLearningMethod.hist_gradient_boosting_regressor:
        regressor = HistGradientBoostingRegressor()
    elif task_parameters.machine_learning_method == MachineLearningMethod.nu_svr:
        regressor = NuSVR()
    elif task_parameters.machine_learning_method == MachineLearningMethod.k_neighbors_regressor:
        regressor = KNeighborsRegressor()
    elif task_parameters.machine_learning_method == MachineLearningMethod.theil_sen_regressor:
        regressor = TheilSenRegressor()
    else:
        raise NotImplementedError

    if task_parameters.meta_regressor == MetaRegressor.bagging_regressor:
        regressor = BaggingRegressor(regressor)
    elif task_parameters.meta_regressor == MetaRegressor.ada_boost_regressor:
        regressor = AdaBoostRegressor(regressor)

    model = make_pipeline(StandardScaler(), regressor)
    model.fit(training_input, training_target)
    prediction: np.ndarray = model.predict(new_input)

    prediction_and_metadata = pd.DataFrame(
        {
            "id": new_ids,
            "prediction": prediction,
            "queue_size": new_queue_sizes
        }
    )
    prediction_and_metadata.sort_values("prediction", ascending=False, inplace=True)

    predictions_with_ids = {}

    for _, circuit in prediction_and_metadata.iterrows():
        predictions_with_ids[circuit["id"]] = circuit["prediction"]

    combined_rank = _calculate_borda_rank(task_parameters, prediction_and_metadata)
    ids_combined_sorted = sort_array_with_ranking(prediction_and_metadata["id"].to_numpy(), combined_rank)

    result = PredictionResult(predictions_with_ids, list(prediction_and_metadata["id"]), list(ids_combined_sorted))

    with SpooledTemporaryFile(mode="wt") as output_file:
        schema = PredictionResultSchema()
        serialized = schema.dumps(result)
        output_file.write(serialized)
        output_file.flush()

        STORE.persist_task_result(
            db_id, output_file, "predictions.json", "text", "application/json"
        )

    return "finished"
