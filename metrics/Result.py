from typing import Any

from pydantic import BaseModel
from datasets import Dataset
from model_interface import Model
from .Generic_Models import *


class Result(BaseModel):
    result_name: str
    storage_location: str
    contained_threshold_optimizer: dict[str, GenericThresholdOptimizer]

    training_dataset: Dataset
    test_dataset: Dataset

    used_model_type: Model
    used_model_name: str = None

    contained_metrics: dict[str, GenericMetric]


class IResultSpec:

    def get_used_model_name(self) -> str:
        raise NotImplementedError
