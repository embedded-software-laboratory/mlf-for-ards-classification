from __future__ import annotations

from typing import Union

from pydantic import BaseModel


class ModelMetadata(BaseModel):
    model_name: str
    model_type: str
    model_algorithm: str

    model_hyperparameters: dict
    model_training_data_location: str
    model_training_evaluation_location: str
    model_storage_location: str


class ModelMetaDataFactory:
    @staticmethod
    def factory_method(model: Union['Model', 'TimeSeriesModel'], training_data_location: str,
                       training_evaluation_location: str) -> ModelMetadata:
        if model.type == "TimeSeriesModel":
            return ModelMetaDataFactory._timeseries_model_factory(model, training_data_location,
                                                                  training_evaluation_location)

    @staticmethod
    def _timeseries_model_factory(model: 'TimeSeriesModel', training_data_location: str,
                                  training_evaluation_location: str) -> ModelMetadata:

        model_hyperparameters = model.get_params()

        return ModelMetadata(model_name=model.name, model_type="TimeSeriesModel",
                             model_hyperparameters=model_hyperparameters, model_storage_location=model.storage_location,
                             model_training_data=training_data_location,
                             model_training_evaluation_location=training_evaluation_location)
