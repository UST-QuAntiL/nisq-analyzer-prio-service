import marshmallow as ma
from marshmallow import EXCLUDE
from qhana_plugin_runner.api.util import FrontendFormBaseSchema


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
