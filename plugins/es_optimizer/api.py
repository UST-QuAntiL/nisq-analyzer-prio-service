from http import HTTPStatus

import marshmallow as ma
from flask.helpers import url_for
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
    description="ES-Optimizer plugin API.",
)


class MetricSchema(FrontendFormBaseSchema):
    class Meta:
        unknown = EXCLUDE

    weight = ma.fields.Float(
        required=True,
        allow_none=False
    )
    is_cost = ma.fields.Boolean(
        required=True,
        allow_none=False
    )


class BordaCountSchema(FrontendFormBaseSchema):
    class Meta:
        unknown = EXCLUDE

    is_cost = ma.fields.Bool(
        required=True,
        allow_none=False
    )


class CircuitSchema(FrontendFormBaseSchema):
    class Meta:
        unknown = EXCLUDE

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
    class Meta:
        unknown = EXCLUDE

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
    class Meta:
        unknown = EXCLUDE

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
    class Meta:
        unknown = EXCLUDE

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
