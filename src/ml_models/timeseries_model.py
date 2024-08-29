from sklearn.metrics import roc_curve
import pydantic

from ml_models.model_interface import Model
from ml_models.ModelMetaData import ModelMetaDataFactory


class TimeSeriesModel(Model):

    def __init__(self):
        super().__init__()
        self.type = 'TimeSeriesModel'

    def get_params(self):
        raise NotImplementedError

    def save(self, filepath, training_dataset_location: str):
        if not self.trained:
            print("It makes no sense to save the model before training it")
            return

        model_meta_data = ModelMetaDataFactory.factory_method(model=self.model,
                                                              training_data_location=training_dataset_location)
        self.save_model(filepath)
        meta_data_path = filepath + "_meta_data.json"
        with open(meta_data_path, "w", encoding="utf-8") as meta_data_file:
            meta_data_file.write(model_meta_data.to_json(indent=4))

    def load(self, filepath, model_name):
        self.load_model(filepath + model_name)
        meta_data_path = filepath + model_name + "_meta_data.json"
        model_meta_data = pydantic.parse_raw(meta_data_path)


        pass


class TimeSeriesProbaModel(TimeSeriesModel):

    def __init__(self):
        super().__init__()

    def has_predict_proba(self):
        return True
