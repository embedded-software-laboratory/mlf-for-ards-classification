

from ml_models.model_interface import Model
from ml_models.ModelMetaData import ModelMetaDataFactory, ModelMetadata

from processing import TimeSeriesDataset
from metrics import EvalResultFactory, EvalResult

from evaluation import ModelEvaluation

import json

class TimeSeriesModel(Model):

    def __init__(self):
        super().__init__()
        self.type = 'TimeSeriesModel'

    def get_params(self):
        raise NotImplementedError

    def set_params(self, params: dict):
        raise NotImplementedError

    @property
    def storage_location(self):
        return self._storage_location

    @storage_location.setter
    def storage_location(self, location):
        self._storage_location = location

    def train_timeseries(self, training_dataset: TimeSeriesDataset,  config, stage: str, split_name: str = " split"):
        model_evaluator = ModelEvaluation(config, self, None)
        training_data = training_dataset.content
        training_data_meta_data = training_dataset.meta_data
        self.train_model(training_data)
        self.meta_data = ModelMetaDataFactory.factory_method(self, training_data_meta_data.dataset_location, training_data_meta_data.dataset_location)
        labels = training_data["ards"]
        predictors = training_data.loc[:, training_data.columns != 'ards']
        model_evaluator.evaluate_timeseries_model(predictors, labels, stage, training_data_meta_data, training_data_meta_data, split_name)

        self.training_evaluation = model_evaluator.evaluation_results[stage]



    def save(self, filepath: str, name: str= None):
        if not self.trained:
            print("It makes no sense to save the model before training it")
            return
        if not name:
            base_path = filepath + f"{self.algorithm}_{self.name}"
        else:
            base_path = filepath + name
        evaluation_location = base_path + "_training_evaluation.json"
        with open(evaluation_location, "w") as evaluation_file:
            print(self.training_evaluation)
            evaluation_file.write(self.training_evaluation.model_dump_json(indent=4))

        self.meta_data.ml_model_training_evaluation_location = evaluation_location
        meta_data_path = base_path + "_meta_data.json"
        with open(meta_data_path, "w", encoding="utf-8") as meta_data_file:
            meta_data_file.write(self.meta_data.model_dump_json(indent=4))

        self.save_model(filepath)

    def load(self, filepath):
        self.load_model(filepath)
        meta_data_path = filepath  + f"{self.algorithm}_{self.name}_meta_data.json"
        with open(meta_data_path, 'r') as meta_data_file:
            meta_data = json.load(meta_data_file)

        model_meta_data = ModelMetadata(**meta_data)
        self.meta_data = model_meta_data

        evaluation_location = filepath + f"{self.algorithm}_{self.name}_training_evaluation.json"
        with open(evaluation_location, 'r') as evaluation_file:
            training_metrics = json.load(evaluation_file)
        model_training_evaluation = EvalResultFactory.from_dict(training_metrics)

        self.trained = True
        self.training_evaluation = model_training_evaluation



class TimeSeriesProbaModel(TimeSeriesModel):

    def __init__(self):
        super().__init__()

    def has_predict_proba(self):
        return True
