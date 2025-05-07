import yaml
import logging

from ml_models.timeseries_model import TimeSeriesModel
from ml_models.adaboost import AdaBoostModel
from ml_models.lightGBM import LightGBMModel
from ml_models.logistic_regression import LogisticRegressionModel
from ml_models.random_forest import RandomForestModel
#from ml_models.recurrentneuralnetworkmodel import RecurrentNeuralNetworkModel
from ml_models.support_vector_machine import SupportVectorMachineModel
from ml_models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)


class TimeSeriesModelManager:

    def __init__(self, config, path):
        self.config = config
        self.save_models = config["process"]["save_models"]
        self.outdir = path

    def create_model_from_config(self, needed_models: dict, base_config_path: str ):
        models = {}
        for model_type in needed_models:
            names = needed_models[model_type]["Names"]
            configs = needed_models[model_type]["Configs"]
            for i in range(len(names)):
                model = eval(model_type+ "Model()")
                model.name = names[i]

                if configs[i] != "default":
                    hyperparameters_path = base_config_path + str.replace(model_type, "Model", "") + "/" + configs[i]
                    with open(hyperparameters_path, 'r') as f:
                        hyperparameters = yaml.safe_load(f)
                    model.set_params(hyperparameters)

                if self.save_models:
                    model.storage_location = f"{self.outdir + model.algorithm}_{model.name}"
                else:
                    model.storage_location = "Model is not saved"

                if model_type in models:
                    models[model_type].append(model)
                else:
                    models[model_type] = [model]
        return models

    def load_models(self, needed_models: dict, available_models_dict: dict, model_base_paths: dict) -> dict:
        for model_type, model_names in needed_models.items():
            available_models = available_models_dict[model_type]
            base_path = model_base_paths[model_type] if model_base_paths[model_type] != "default" else self.outdir
            for model_name in model_names["Names"]:
                found = False
                for model in available_models:
                    if model_name == model.name:
                        found = True
                if not found:
                    logger.info(f"Model {model_name} of type {model_type} not found. Loading model from disk...")
                    model = eval(model_type + "Model()")
                    model.name = model_name
                    model.algorithm = model_type
                    model.load(base_path)
                    available_models_dict[model_type].append(model)
                    logger.info(f"Loaded model {model_name} of type {model_type}")
        return available_models_dict