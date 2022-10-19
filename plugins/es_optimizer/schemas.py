from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

import marshmallow as ma
from marshmallow import EXCLUDE, post_load
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
            MetricSchema()
        ),
        required=True,
        allow_none=False,
    )
    borda_count_metrics = ma.fields.Dict(
        keys=ma.fields.String(),
        values=ma.fields.Nested(
            BordaCountSchema()
        ),
        required=True,
        allow_none=False
    )
    circuits = ma.fields.List(
        ma.fields.Nested(CircuitSchema()),
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
            MetricSchema()
        ),
        required=True,
        allow_none=False,
    )
    circuits = ma.fields.List(
        ma.fields.Nested(CircuitSchema()),
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
            MetricSchema()
        ),
        required=True,
        allow_none=False,
    )
    borda_count_metrics = ma.fields.Dict(
        keys=ma.fields.String(),
        values=ma.fields.Nested(
            BordaCountSchema()
        ),
        required=True,
        allow_none=False
    )
    circuits = ma.fields.List(
        ma.fields.Nested(CircuitSchema()),
        required=True,
        allow_none=False
    )


class MachineLearningMethod(Enum):
    extra_trees_regressor = "extra_trees_regressor"
    gradient_boosting_regressor = "gradient_boosting_regressor"
    random_forest_regressor = "random_forest_regressor"


@dataclass
class OriginalCircuit:
    id: str
    original_circuit_and_qpu_metrics: List[Dict[str, Any]]


class OriginalCircuitSchema(FrontendFormBaseSchema):
    class Meta:
        unknown = EXCLUDE

    id = ma.fields.String(
        required=True,
        allow_none=False
    )
    original_circuit_and_qpu_metrics = ma.fields.List(
        ma.fields.Dict(
            keys=ma.fields.String()
        ),
        required=True,
        allow_none=False
    )

    @post_load
    def make_object(self, data, **kwargs):
        return OriginalCircuit(**data)


@dataclass
class LearnPrediction:
    machine_learning_method: MachineLearningMethod
    training_data: List[OriginalCircuit]
    new_circuit: OriginalCircuit
    input_metric_names: List[str]
    compiler_property_name: str
    histogram_intersection_name: str
    queue_size_name: str
    queue_size_importance: float


class LearnPredictionSchema(FrontendFormBaseSchema):
    class Meta:
        unknown = EXCLUDE

    machine_learning_method = ma.fields.Enum(
        MachineLearningMethod,
        required=True,
        allow_none=False
    )
    training_data = ma.fields.List(
        ma.fields.Nested(OriginalCircuitSchema()),
        required=True,
        allow_none=False
    )
    new_circuit = ma.fields.Nested(
        OriginalCircuitSchema(),
        required=True,
        allow_none=False
    )
    input_metric_names = ma.fields.List(
        ma.fields.String(),
        required=True,
        allow_none=False
    )
    compiler_property_name = ma.fields.Str(
        required=True,
        allow_none=False
    )
    histogram_intersection_name = ma.fields.Str(
        required=True,
        allow_none=False
    )
    queue_size_name = ma.fields.Str(
        required=True,
        allow_none=False
    )
    # 0.0 = ignore queue size, 0.5 = queue size as important as predicted histogram intersection,
    # 1.0 = only consider queue size
    queue_size_importance = ma.fields.Float(
        required=True,
        allow_none=False
    )

    @post_load
    def make_object(self, data, **kwargs):
        return LearnPrediction(**data)


@dataclass
class PredictionResult:
    predicted_histogram_intersections: Dict[str, float]
    ranking: List[str]
    borda_count_ranking: List[str]


class PredictionResultSchema(FrontendFormBaseSchema):
    class Meta:
        unknown = EXCLUDE

    predicted_histogram_intersections: ma.fields.Dict(
        keys=ma.fields.String(),
        values=ma.fields.Float(),
        required=True,
        allow_none=False
    )
    original_circuit_and_qpu_metrics = ma.fields.List(
        ma.fields.Dict(
            keys=ma.fields.String()
        ),
        required=True,
        allow_none=False
    )

    @post_load
    def make_object(self, data, **kwargs):
        return OriginalCircuit(**data)
