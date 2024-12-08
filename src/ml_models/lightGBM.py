from lightgbm import LGBMClassifier
import pickle

from ml_models.timeseries_model import TimeSeriesProbaModel
from ml_models.model_interface import Model
from ml_models.timeseries_model import TimeSeriesModel

class LightGBMModel(TimeSeriesProbaModel):
    def __init__(self):
        super().__init__()

        self.name = "LightGBMModel"
        self.algorithm = "LightGBM"

        self.hyperparameters = {
            "boosting_type": 'gbdt',
            "num_leaves": 31,
            "max_depth": -1,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample_for_bin": 500,
            "objective": 'binary',
            "class_weight": None,
            "min_split_gain": 0.0,
            "min_child_weight": 0.001,
            "min_child_samples": 20,
            "subsample": 1.0,
            "subsample_freq": 0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "random_state": 0,
            "n_jobs": -1,
            "importance_type": 'split',
            "metric": 'binary_error',
            "is_unbalance": True,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": 0
        }







        #What about early stopping round als attribut wenn das Model eskaliert?
        self.model = self._init_gbm()

        #klappt das?

    def train_model(self, training_data):
        # Daten und Labels extrahieren
        label = training_data["ards"]
        predictors = training_data.loc[:, training_data.columns != 'ards']

        self.model = self.model.fit(predictors, label)
        self.trained = True

    def predict(self, data):
        # Vorhersage für einen einzelnen Patienten machen
        return self.model.predict(data)

    def predict_proba(self, data):
        # Vorhersage der Wahrscheinlichkeiten für die Klassenzuordnung machen
        return self.model.predict_proba(data)

    def _init_gbm(self) -> LGBMClassifier:
        lightGBM = LGBMClassifier()
        lightGBM.set_params(**self.hyperparameters)
        return lightGBM

    def get_params(self):
        return self.model.get_params(True)

    def set_params(self, params):
        for key, value in params.items():
            if key in self.hyperparameters:
                self.hyperparameters[key] = value
        self.model.set_params(**self.hyperparameters)

    def save_model(self, filepath):
        file = open(filepath + f"{self.algorithm}_{self.name}.pkl", "wb")
        pickle.dump(self.model, file)

    def load_model(self, filepath):
        file = open(filepath + f"{self.algorithm}_{self.name}.pkl", "rb")
        self.model = pickle.load(file)

    def has_predict_proba(self):
        return True


#LightGBMModel().save("./Save/LightGBMModel")

# Laden
#LightGBMModel().load("./Save/LightGBMModel")
