from __future__ import annotations

from typing import Any, Callable, Union
import numpy as np

from pydantic import BaseModel


class GenericSplit(BaseModel):
    split_name: str
    contained_optimization: dict[str, GenericMetric]


class GenericThresholdOptimization(BaseModel):
    optimization_name: str
    contained_metrics: dict[str, GenericSplit]


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


class IMetricSpec:

    def calculate_metric(self, metric_parameters: dict) -> GenericValue:
        raise NotImplementedError

    def calculate_metric_mean(self, average_parmeters: dict) -> GenericValue:
        raise NotImplementedError

    def needs_probabilities(self) -> bool:
        raise NotImplementedError


class IFloatMetricSpec(IMetricSpec):

    def __init__(self):
        super().__init__()
        self.metric_spec = IFloatMetricSpec
        self.metric_type = GenericMetric

    def calculate_metric(self, metric_parameters: dict) -> FloatValue:
        raise NotImplementedError

    def calculate_metric_mean(self, average_parameters: dict) -> FloatValue:
        metric_value_sum = 0.0
        for _, value in average_parameters.items():
            metric_value_sum += value
        return FloatValue(metric_value=metric_value_sum / len(average_parameters))


class IIntMetricSpec(IMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> IntValue:
        raise NotImplementedError

    def calculate_metric_mean(self, average_parameters: dict) -> FloatValue:
        metric_value_sum = 0.0
        for _, value in average_parameters.items():
            metric_value_sum += value
        return FloatValue(metric_value=metric_value_sum / len(average_parameters))


class IListMetricSpec(IMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> ListValue:
        raise NotImplementedError

    def calculate_metric_mean(self, average_parameters: dict) -> ListValue:
        return ListValue(metric_value=["Mean calculation makes no sense"])
