from __future__ import annotations

from pydantic import BaseModel, ValidationInfo, field_validator, model_serializer, model_validator
from typing import Any,  Union

from processing import TimeseriesMetaData
from processing.datasets_metadata import ImageMetaData


class GenericSplit(BaseModel):
    split_name: str
    contained_metrics: dict[str, GenericMetric]

    class Config:
        arbitrary_types_allowed = True


class GenericThresholdOptimization(BaseModel):
    optimization_name: str
    contained_splits: dict[str, GenericSplit]

    class Config:
        arbitrary_types_allowed = True


class GenericMetric(BaseModel):
    metric_name: str
    metric_value: GenericValue
    metric_spec: IMetricSpec

    @model_serializer()
    def serialize(self):
        return {"metric_name": self.metric_name, "metric_value": self.metric_value.metric_value, "metric_spec": self.metric_spec.__class__.__name__}

    def __lt__(self, other):
        return self.metric_value < other

    class Config:
        arbitrary_types_allowed = True


class GenericValue(BaseModel):
    metric_value: Union[ListValue, IntValue, FloatValue, StringValue]

    def __lt__(self, other):
        return self.metric_value < other

    def __gt__(self, other):
        return self.metric_value > other

    class Config:
        arbitrary_types_allowed = True


class ListValue(GenericValue):
    metric_value: list[Union[int, float, str]]


class IntValue(GenericValue):
    metric_value: int


class FloatValue(GenericValue):
    metric_value: float


class StringValue(GenericValue):
    metric_value: str


class ExperimentResult(BaseModel):
    result_name: str
    storage_location: str
    result_version: str = "1.0"
    contained_model_results: dict

    class Config:
        arbitrary_types_allowed = True


class ModelResult(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    used_model_location: str
    used_model_name: str = None
    used_model_algorithm: str = None
    used_model_type: str = None
    contained_evals: dict


class EvalResult(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    eval_type: str

    training_dataset: Union[TimeseriesMetaData, ImageMetaData, None] = None
    test_dataset: Union[TimeseriesMetaData, ImageMetaData, None] = None

    contained_optimizers: dict[str, GenericThresholdOptimization]

    crossvalidation_performed: bool
    crossvalidation_random_state: Union[int, None] = None
    crossvalidation_shuffle: Union[bool, None] = None
    crossvalidation_splits: Union[int, None] = None
    evaluation_performed: bool

    @field_validator('crossvalidation_random_state', 'crossvalidation_splits')
    @classmethod
    def check_crossvalidation_settings_int(cls, v: int, info: ValidationInfo):
        if info.data['crossvalidation_performed']:
            if isinstance(v, int):
                assert v is not None, f'{info.field_name} must be set if crossvalidation_performed is set to True'
                assert v >= 0, f'{info.field_name} must be greater than zero if crossvalidation_performed is set to True'
        return v

    @field_validator('crossvalidation_shuffle')
    @classmethod
    def check_crossvalidation_shuffle_settings_bool(cls, v: bool, info: ValidationInfo):
        if info.data['crossvalidation_performed']:
            if isinstance(v, bool):
                assert v is not None, f'{info.field_name} must be set if crossvalidation_performed is set to True'
        return v

    @model_validator(mode="after")
    def check_only_one_eval_cross_val(self):
        if not  (self.crossvalidation_performed or self.evaluation_performed):
            raise ValueError("Either CV or Eval has to be performed")
        if (self.crossvalidation_performed and self.evaluation_performed):
            raise ValueError("CV and Eval can not be performed in the same Evaluation")
        return self








class IMetricSpec:

    def calculate_metric(self, metric_parameters: dict, stage: str) -> GenericValue:
        raise NotImplementedError

    def calculate_metric_mean(self, average_parmeters: dict) -> GenericValue:
        raise NotImplementedError

    def needs_probabilities(self) -> bool:
        raise NotImplementedError

    def create_from_value(self, metric_value: GenericValue, metric_name: str) -> GenericMetric:
        raise NotImplementedError

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        raise NotImplementedError


class FloatMetricSpec(IMetricSpec):

    def __init__(self):
        super().__init__()
        self.metric_spec = FloatMetricSpec
        self.metric_type = GenericMetric

    def calculate_metric(self, metric_parameters: dict, stage:str) -> FloatValue:
        raise NotImplementedError

    def calculate_metric_mean(self, average_parameters: list) -> FloatValue:
        metric_value_sum = 0.0
        for value in average_parameters:
            metric_value_sum += value.metric_value.metric_value

        return FloatValue(metric_value=metric_value_sum / len(average_parameters))


class IntMetricSpec(IMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> IntValue:
        raise NotImplementedError

    def calculate_metric_mean(self, average_parameters: list) -> FloatValue:
        metric_value_sum = 0
        for value in average_parameters:
            metric_value_sum += value.metric_value.metric_value
        return FloatValue(metric_value=metric_value_sum / len(average_parameters))


class ListMetricSpec(IMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> ListValue:
        raise NotImplementedError

    def calculate_metric_mean(self, average_parameters: list) -> ListValue:
        return ListValue(metric_value=["Mean calculation makes no sense"])


