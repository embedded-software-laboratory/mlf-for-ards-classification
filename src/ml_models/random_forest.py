from ml_models.timeseries_model import TimeSeriesProbaModel
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np


class RandomForestModel(TimeSeriesProbaModel):

    def __init__(self):
        super().__init__()
        self.name = "Random Forest"
        self.algorithm = "RandomForest"

        self.hyperparameters = {
            "n_estimators": 700,
            "criterion": "gini",
            "max_depth": 200,
            "min_samples_split": 10,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_features": "sqrt",
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "bootstrap": True,
            "oob_score": False,
            "n_jobs": -1,
            "random_state": 42,
            "verbose": 0,
            "warm_start": False,
            "class_weight": None,
            "ccp_alpha": 0.0,
            "max_samples": None,
        }

        self.model = self._init_forest()

    def train_model(self, training_data):
        """Function that starts the learning process of the RF and stores the resulting model after completion"""

        # Init forest and read training data
        label = training_data["ards"]
        
        # Convert to standard numpy array to avoid pandas dtype issues
        label = label.astype(int).values
        
        predictors = training_data.loc[:, training_data.columns != 'ards']

        # Learn and store resulting model
        self.model = self.model.fit(predictors, label)
        print(self.model.feature_importances_)
        self.trained = True

    def predict(self, data):
        return self.model.predict(data)

    def predict_proba(self, data):
        return self.model.predict_proba(data)

    def _init_forest(self) -> RandomForestClassifier:
        """Function that intializes the Random Forest"""

        # Init RF
        random_forest = RandomForestClassifier()
        random_forest.set_params(**self.hyperparameters)
        return random_forest

    def get_params(self):
        return self.model.get_params(deep=True)

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


