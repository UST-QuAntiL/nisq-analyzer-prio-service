from http import HTTPStatus
from typing import Any, Mapping, Optional, List, Dict, Tuple

import marshmallow as ma
from flask.wrappers import Response
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
)
from qhana_plugin_runner.api.util import FrontendFormBaseSchema, SecurityBlueprint

from .plugin import EsOptimizer

# Blueprint to register API endpoints with
PLUGIN_BLP = SecurityBlueprint( #SecurityBlueprint for eventual JWT support
    EsOptimizer.instance.identifier,  # blueprint name
    __name__,  # module import name!
    description="An optimizer using an evolutionary strategy.",
)


class MetricSchema(FrontendFormBaseSchema):
    weight = ma.fields.Float(
        required=True,
        allow_none=False
    )
    is_cost = ma.fields.Boolean(
        required=True,
        allow_none=False
    )


class BordaCountSchema(FrontendFormBaseSchema):
    is_cost = ma.fields.Bool(
        required=True,
        allow_none=False
    )


class CircuitSchema(FrontendFormBaseSchema):
    id = ma.fields.String(
        required=True,
        allow_none=False
    )
    compiled_circuits = ma.fields.List(
        ma.fields.Dict(
            keys=ma.fields.String()
        ),
        required=True,
        allow_none=False
    )


class RankSchema(FrontendFormBaseSchema):
    mcda_method = ma.fields.String(
        required=True,
        allow_none=False
    )
    metric_weights = ma.fields.Dict(
        keys=ma.fields.String(),
        values=ma.fields.Nested(
            MetricSchema
        ),
        required=True,
        allow_none=False,
    )
    borda_count_metrics = ma.fields.Dict(
        keys=ma.fields.String(),
        values=ma.fields.Nested(
            BordaCountSchema
        ),
        required=True,
        allow_none=False
    )
    circuits = ma.fields.List(
        ma.fields.Nested(CircuitSchema),
        required=True,
        allow_none=False
    )


class LearnRankingSchema(FrontendFormBaseSchema):
    mcda_method = ma.fields.String(
        required=True,
        allow_none=False
    )
    learning_method = ma.fields.String(
        required=True,
        allow_none=False
    )
    metric_weights = ma.fields.Dict(
        keys=ma.fields.String(),
        values=ma.fields.Nested(
            MetricSchema
        ),
        required=True,
        allow_none=False,
    )
    circuits = ma.fields.List(
        ma.fields.Nested(CircuitSchema),
        required=True,
        allow_none=False
    )


class RankSensitivitySchema(FrontendFormBaseSchema):
    mcda_method = ma.fields.String(
        required=True,
        allow_none=False
    )
    step_size = ma.fields.Float(
        required=True,
        allow_none=False
    )
    upper_bound = ma.fields.Float(
        required=True,
        allow_none=False
    )
    lower_bound = ma.fields.Float(
        required=True,
        allow_none=False
    )
    metric_weights = ma.fields.Dict(
        keys=ma.fields.String(),
        values=ma.fields.Nested(
            MetricSchema
        ),
        required=True,
        allow_none=False,
    )
    borda_count_metrics = ma.fields.Dict(
        keys=ma.fields.String(),
        values=ma.fields.Nested(
            BordaCountSchema
        ),
        required=True,
        allow_none=False
    )
    circuits = ma.fields.List(
        ma.fields.Nested(CircuitSchema),
        required=True,
        allow_none=False
    )


@PLUGIN_BLP.route("/")
class PluginView(MethodView):
    """Root resource of this plugin."""

    @PLUGIN_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @PLUGIN_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            # human readable title and description
            title=EsOptimizer.instance.name,
            description=PLUGIN_BLP.description,
            # machine readable identifying name and version
            name=EsOptimizer.instance.identifier,
            version=EsOptimizer.instance.version,
            # pluin type: "processing"|"visualizing"|"conversion" (actual values still WIP!! confirm with current documentation)
            type=PluginType.simple,
            # tags describing the plugin, e.g. ml:autoencoder, ml:svm
            tags=[],
            # the main plugin entry point
            entry_point=EntryPoint(
                # entry point for headless (non-gui) applications
                href=url_for(f"{PLUGIN_BLP.name}.ProcessView"),
                # micro frontend entry point
                ui_href=url_for(f"{PLUGIN_BLP.name}.MicroFrontend"),
                # definition of (required) input data
                data_input=[],
                # definition of output data
                data_output=[
                    DataMetadata(
                        data_type="txt",
                        content_type=["text/plain"],
                        required=True,
                    )
                ],
            ),
        )


# @PLUGIN_BLP.route("/ui/")
# class MicroFrontend(MethodView):
#     """Micro frontend for ES Optimizer."""
#
#     @PLUGIN_BLP.html_response(
#         HTTPStatus.OK, description="Micro frontend of ES Optimizer."
#     )
#     @PLUGIN_BLP.arguments(
#         EsOptimizerParametersSchema(
#             # partial to allow partial inputs, validate as errors to get all validation errors
#             # data is still stored in request.args
#             partial=True, unknown=EXCLUDE, validate_errors_as_result=True
#         ),
#         location="query", # allow get submit via query parameters
#         required=False, # micro frontend should just ignore most errors (but return error messages!)
#     )
#     @PLUGIN_BLP.require_jwt("jwt", optional=True)
#     def get(self, errors):
#         """Return the micro frontend."""
#         return self.render(request.args, errors)
#
#     @PLUGIN_BLP.html_response(
#         HTTPStatus.OK, description="Micro frontend of the hello world plugin."
#     )
#     @PLUGIN_BLP.arguments(
#         EsOptimizerParametersSchema(
#             # partial to allow partial inputs, validate as errors to get all validation errors
#             # data is still stored in request.form
#             partial=True, unknown=EXCLUDE, validate_errors_as_result=True
#         ),
#         location="form", # allow post submit of html form
#         required=False, # micro frontend should just ignore most errors (but return error messages!)
#     )
#     @PLUGIN_BLP.require_jwt("jwt", optional=True)
#     def post(self, errors):
#         """Return the micro frontend with prerendered inputs."""
#         return self.render(request.form, errors)
#
#     def render(self, data: Mapping, errors: dict):
#         """Render the micro frontend from the html template.
#
#         Args:
#             data (Mapping): The submitted data (may be incomplete!)
#             errors (dict): Validation errors with the current submitted data
#
#         Returns:
#             Response: the rendered template with the submitted data prefilled and errors
#         """
#         schema = EsOptimizerParametersSchema()
#         return Response(
#             render_template(
#                 "simple_template.html",
#                 name=EsOptimizer.instance.name,
#                 version=EsOptimizer.instance.version,
#                 schema=schema, #schema is used to create the html form
#                 values=data, # data is used to prefill values in the form
#                 errors=errors, # errors is used to show error texts in the form
#                 process=url_for(f"{PLUGIN_BLP.name}.ProcessView"), # the endpoint starting the background task
#             )
#         )
