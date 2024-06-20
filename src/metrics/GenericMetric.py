from pydantic import BaseModel
from Generic_Models import *
from abc import ABC, abstractmethod


class GenericMetric(BaseModel):
    metric_name: str
    metric_calc_func: Callable[[dict[str, GenericValue]], GenericValue]
    metric_mean_func: Callable[[list[GenericMetric]], GenericMetric] = None
    needed_parameters: dict[str, GenericValue]




