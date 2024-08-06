from __future__ import annotations

from pydantic import BaseModel, ValidationInfo, field_validator, ConfigDict, field_serializer


from ml_models.model_interface import Model

from typing import Any, Callable, Union


class GenericSplit(BaseModel):
    split_name: str
    contained_metrics: dict[str, GenericMetric]
    


class GenericThresholdOptimization(BaseModel):
    optimization_name: str
    contained_splits: dict[str, GenericSplit]


class GenericMetric(BaseModel):
    metric_name: str
    metric_value: GenericValue
    metric_spec: IMetricSpec


    def __lt__(self, other):
        return self.metric_value < other

    class Config:
        arbitrary_types_allowed = True


class GenericValue(BaseModel):
    metric_value: Union[ListValue, IntValue, FloatValue, StringValue]

    def __lt__(self, other):
        return self.metric_value < other


class ListValue(GenericValue):
    metric_value: list[Any]


class IntValue(GenericValue):
    metric_value: int


class FloatValue(GenericValue):
    metric_value: float


class StringValue(GenericValue):
    metric_value: str


class Result(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    result_name: str
    storage_location: str

    training_dataset: object = None  # TODO add data set information
    test_dataset: object = None  # TODO add data set information

    used_model_type: Model
    used_model_name: str = None
    contained_optimizers: dict[str, GenericThresholdOptimization]

    crossvalidation_performed: bool
    crossvalidation_random_state: int = None
    crossvalidation_shuffle: bool = None
    crossvalidation_splits: int = None
    evaluation_performed: bool

    @field_validator('crossvalidation_random_state', 'crossvalidation_splits')
    @classmethod
    def check_crossvalidation_settings_int(cls, v: int, info: ValidationInfo):
        if info.data['crossvalidation_performed']:
            if isinstance(v, int):
                assert v is not None, f'{info.field_name} must be set if crossvalidation_performed is set to True'
                assert v >= 0, f'{info.field_name} must be greater than zero if crossvalidation_performed is set to False'
        return v

    @field_validator('crossvalidation_shuffle', )
    @classmethod
    def check_crossvalidation_shuffle_settings_bool(cls, v: bool, info: ValidationInfo):
        if info.data['crossvalidation_performed']:
            if isinstance(v, bool):
                assert v is not None, f'{info.field_name} must be set if crossvalidation_performed is set to True'
        return v

    # TODO read model when reading model from json
    @field_serializer('used_model_type')
    def serialize_model(used_model_type: Model):
        print("Storage serialized")
        return used_model_type.storage_location


class IMetricSpec:

    def calculate_metric(self, metric_parameters: dict) -> GenericValue:
        raise NotImplementedError

    def calculate_metric_mean(self, average_parmeters: dict) -> GenericValue:
        raise NotImplementedError

    def needs_probabilities(self) -> bool:
        raise NotImplementedError


class FloatMetricSpec(IMetricSpec):

    def __init__(self):
        super().__init__()
        self.metric_spec = FloatMetricSpec
        self.metric_type = GenericMetric

    def calculate_metric(self, metric_parameters: dict) -> FloatValue:
        raise NotImplementedError

    def calculate_metric_mean(self, average_parameters: list) -> FloatValue:
        metric_value_sum = 0.0
        for value in average_parameters:
            metric_value_sum += value.metric_value.metric_value
        return FloatValue(metric_value=metric_value_sum / len(average_parameters))


class IntMetricSpec(IMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> IntValue:
        raise NotImplementedError

    def calculate_metric_mean(self, average_parameters: list) -> FloatValue:
        metric_value_sum = 0
        for value in average_parameters:
            metric_value_sum += value.metric_value.metric_value
        return FloatValue(metric_value=metric_value_sum / len(average_parameters))


class ListMetricSpec(IMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> ListValue:
        raise NotImplementedError

    def calculate_metric_mean(self, average_parameters: list) -> ListValue:
        return ListValue(metric_value=["Mean calculation makes no sense"])

