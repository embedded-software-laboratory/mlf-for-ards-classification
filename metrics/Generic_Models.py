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
    metric_calc_func: Callable
    metric_mean_func: Callable[[list[GenericMetric]], GenericMetric] = None
    metric_value: GenericMetricValue



class GenericMetricValue(BaseModel):
    metric_value: ListMetricValue | IntMetricValue | FloatMetricValue


class ListMetricValue(GenericMetricValue):
    metric_value: list[Any]


class IntMetricValue(GenericMetricValue):
    metric_value: int


class FloatMetricValue(GenericMetricValue):
    metric_value: float


class GenericMetricOld:
    def __init__(self, threshold_calc: str, metric_name: str, split: Any, values: Any = None):
        self._threshold_calc = threshold_calc
        self._split = split
        self._name = metric_name

        if values is None:
            self._values = []
        else:
            if isinstance(values, np.ndarray):
                self._values = values.tolist()
            else:
                self._values = values

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, value):
        self._split = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, value):
        if isinstance(value, np.ndarray):
            self._values = value.tolist()
        else:
            self._values = value

    def to_dict(self) -> (str, dict):
        generic_metric_dict = {self.split: self._values}
        return self._name, self._threshold_calc, generic_metric_dict
