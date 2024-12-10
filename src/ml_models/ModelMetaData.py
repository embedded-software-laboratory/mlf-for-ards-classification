from __future__ import annotations

from typing import Union

from pydantic import BaseModel


class ModelMetadata(BaseModel):
    ml_model_name: str
    ml_model_type: str
    ml_model_algorithm: str

    ml_model_hyperparameters: dict
    ml_model_training_data_location: str
    ml_model_training_evaluation_location: str
    ml_model_storage_location: str


class ModelMetaDataFactory:
    @staticmethod
    def factory_method(model: Union['Model', 'TimeSeriesModel'], training_data_location: str,
                    evaluation_data_location: str) -> ModelMetadata:
        if model.type == "TimeSeriesModel":
            return ModelMetaDataFactory._timeseries_model_factory(model, training_data_location,
                                                                  evaluation_data_location)

    @staticmethod
    def _timeseries_model_factory(model: 'TimeSeriesModel', training_data_location: str,
                                  training_evaluation_location: str) -> ModelMetadata:

        ml_model_hyperparameters = model.get_params()

        return ModelMetadata(ml_model_name=model.name, ml_model_type="TimeSeriesModel",
                             ml_model_algorithm=model.algorithm,
                             ml_model_hyperparameters=ml_model_hyperparameters,
                             ml_model_storage_location=model.storage_location,
                             ml_model_training_data_location=training_data_location,
                             ml_model_training_evaluation_location=training_evaluation_location)
