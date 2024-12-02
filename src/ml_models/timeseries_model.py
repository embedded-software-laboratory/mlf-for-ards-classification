
import pydantic

from ml_models.model_interface import Model

from evaluation.Evaluation import ModelEvaluation
from ml_models.ModelMetaData import ModelMetaDataFactory
from processing import TimeSeriesDataset


class TimeSeriesModel(Model):

    def __init__(self):
        super().__init__()
        self.type = 'TimeSeriesModel'

    def get_params(self):
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


    def save(self, filepath):
        if not self.trained:
            print("It makes no sense to save the model before training it")
            return

        evaluation_location = filepath + "_training_evaluation.json"
        with open(evaluation_location, "w") as evaluation_file:
            evaluation_file.write(self.training_evaluation.to_json(indent=4))

        self.meta_data.ml_model_training_evaluation_location = evaluation_location
        meta_data_path = filepath + "_meta_data.json"
        with open(meta_data_path, "w", encoding="utf-8") as meta_data_file:
            meta_data_file.write(self.meta_data.model_dump_json(indent=4))

        self.save_model(filepath)

    def load(self, filepath, model_name):
        # TODO finish
        self.load_model(filepath + model_name)
        meta_data_path = filepath + model_name + "_meta_data.json"
        model_meta_data = pydantic.parse_raw(meta_data_path)

        pass


class TimeSeriesProbaModel(TimeSeriesModel):

    def __init__(self):
        super().__init__()

    def has_predict_proba(self):
        return True
