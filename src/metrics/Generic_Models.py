from __future__ import annotations

from typing import Any, Callable
import numpy as np

from pydantic import BaseModel


class GenericSplit(BaseModel):
    split_name: str
    contained_metrics: list[GenericMetric]


class GenericThresholdOptimizer(BaseModel):
    threshold_optimizer_name: str
    contained_splits: list[GenericSplit]
    threshold_calc_func: Callable


class GenericMetric(BaseModel):
    metric_name: str
    metric_calc_func: Callable[[dict[str, GenericValue]], GenericValue]
    metric_mean_func: Callable[[list[GenericMetric]], GenericMetric] = None
    metric_value: GenericValue
    needed_parameters: dict[str, GenericValue]


class GenericValue(BaseModel):
    metric_value: ListValue | IntValue | FloatValue


class ListValue(GenericValue):
    metric_value: list[Any]


class IntValue(GenericValue):
    metric_value: int


class FloatValue(GenericValue):
    metric_value: float


class StringValue(GenericValue):
    metric_value: str
