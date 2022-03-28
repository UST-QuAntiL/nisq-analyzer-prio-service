import json
import math
from datetime import datetime
from http import HTTPStatus
from json import dumps, loads
from tempfile import SpooledTemporaryFile
from typing import Optional, Dict, Any, List

from celery import chain
from celery.utils.log import get_task_logger
from flask import redirect, url_for
from flask.views import MethodView
from marshmallow import EXCLUDE
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_result, save_task_error

from ..api import PLUGIN_BLP, RankSensitivitySchema
from ..borda_count import borda_count_rank
from ..plugin import EsOptimizer


@PLUGIN_BLP.route("/rank-sensitivity")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @PLUGIN_BLP.arguments(RankSensitivitySchema(unknown=EXCLUDE), location="json")
    @PLUGIN_BLP.response(HTTPStatus.SEE_OTHER)
    @PLUGIN_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the background task."""
        # create a new task instance in DB with the relevant parameters
        db_task = ProcessingTask(task_name=rank_sensitivity_task.name, parameters=dumps(arguments))
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = rank_sensitivity_task.s(db_id=db_task.id) | save_task_result.s(
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


def replace_nan_with_none(float_list: List[float]) -> List[float]:
    new_list = []

    for f in float_list:
        if math.isnan(f):
            new_list.append(None)
        else:
            new_list.append(f)

    return new_list


# task names must be globally unique => use full versioned plugin identifier to scope name
@CELERY.task(name=f"{EsOptimizer.instance.identifier}.rank_sensitivity_task", bind=True)
def rank_sensitivity_task(self, db_id: int) -> str:
    import numpy as np
    from plotly.subplots import make_subplots
    from pymcdm.methods import TOPSIS, PROMETHEE_II
    import plotly.graph_objects as go
    from ..parsing import get_metrics_from_compiled_circuits, parse_metric_info, get_rankings_for_borda_count
    from ..sensitivity import find_changing_factors
    from ..tools.ranking import convert_scores_to_ranking, sort_array_with_ranking
    from ..weights import NormalizedWeights

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

    if task_parameters["mcda_method"] == "topsis":
        mcda = TOPSIS()
    elif task_parameters["mcda_method"] == "promethee_ii":
        mcda = PROMETHEE_II("usual")
    else:
        raise ValueError("Unknown method: " + str(task_parameters["method"]))

    original_scores: np.ndarray = mcda(metrics, weights, is_cost)
    original_ranking = convert_scores_to_ranking(original_scores, True)
    step_size: float = task_parameters["step_size"]
    upper_bound: float = task_parameters["upper_bound"]
    lower_bound: float = task_parameters["lower_bound"]

    output_data = {
        "original_scores": original_scores.tolist(),
        "original_ranking": original_ranking.tolist()
    }

    rankings_for_borda = get_rankings_for_borda_count(task_parameters, 0)

    if len(rankings_for_borda) > 0:
        borda_rank = borda_count_rank([original_ranking] + rankings_for_borda)

        output_data["original_borda_count_ranking"] = borda_rank.tolist()

    if len(rankings_for_borda) > 0:
        rankings_for_borda = [rankings_for_borda]
    else:
        rankings_for_borda = None

    decreasing_factors, decreasing_ranks, decreasing_borda_ranks, increasing_factors, increasing_ranks, increasing_borda_ranks = find_changing_factors(mcda, [metrics], is_cost, NormalizedWeights(weights), rankings_for_borda, step_size, upper_bound, lower_bound)

    # remove unused dimension
    decreasing_ranks = [dr[0] if len(dr) > 0 else [] for dr in decreasing_ranks]
    increasing_ranks = [ir[0] if len(ir) > 0 else [] for ir in increasing_ranks]

    output_data["decreasing_factors"] = replace_nan_with_none(decreasing_factors)
    output_data["disturbed_ranks_decreased"] = decreasing_ranks

    if rankings_for_borda is not None:
        output_data["disturbed_borda_ranks_decreased"] = decreasing_borda_ranks

    output_data["increasing_factors"] = replace_nan_with_none(increasing_factors)
    output_data["disturbed_ranks_increased"] = increasing_ranks

    if rankings_for_borda is not None:
        output_data["disturbed_borda_ranks_increased"] = increasing_borda_ranks

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig.update_yaxes({"range": [0.9, np.nanmax(increasing_factors) * 1.1]}, row=1)
    fig.update_yaxes({"range": [0, 1.1]}, row=2)

    # create hover text for the plot, disturbed ranking are sorted to make the comparison to the original ranking easier
    decreasing_ranks_text = [str(sort_array_with_ranking(np.array(dr) + 1, original_ranking)) if len(dr) > 0 else "" for dr in decreasing_ranks]
    increasing_ranks_text = [str(sort_array_with_ranking(np.array(ir) + 1, original_ranking)) if len(ir) > 0 else "" for ir in increasing_ranks]

    fig.add_trace(
        go.Scatter(x=metric_names, y=increasing_factors, name="increasing factors", mode="markers", marker={"symbol": "triangle-up", "size": 10}, hovertext=increasing_ranks_text),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=metric_names, y=decreasing_factors, name="decreasing factors", mode="markers", marker={"symbol": "triangle-up", "size": 10}, hovertext=decreasing_ranks_text),
        row=2, col=1
    )

    with SpooledTemporaryFile(mode="wt") as output_file:
        json.dump(output_data, output_file)
        STORE.persist_task_result(
            db_id, output_file, "sensitivity.json", "text", "application/json"
        )

    with SpooledTemporaryFile(mode="wt") as output_file:
        fig.write_html(output_file)
        STORE.persist_task_result(
            db_id, output_file, "plot.html", "plot", "text/html"
        )

    return "finished"
