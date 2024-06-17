from typing import Any

from pydantic import BaseModel
from datasets import Dataset
from model_interface import Model
from .Generic_Metric import GenericMetric


class Result(BaseModel):


    result_name: str
    storage_location: str
    contained_metrics: dict[str, GenericMetric]

    training_dataset: Dataset
    test_dataset: Dataset

    used_model_type: Model
    used_model_name: str = None

    contained_metrics: dict[str, GenericMetric]

    def contained_metrics(self) -> dict[str, GenericMetric]:
        return self.contained_metrics
    def update(self):


    def name(self) -> str:
        return self.result_name




class IResultSpec:

    def get_used_model_name(self) -> str:
        raise NotImplementedError
