from __future__ import annotations
from abc import ABC, abstractmethod

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




class ResultFactory:


    def factory_method(self, result_config:dict) -> Result:
        threshold_optimizers = result_config['used_threshold_optimizers']
        number_of_splits = result_config['number_of_splits']
        contained_metrics = result_config['metrics']
        
        pass



