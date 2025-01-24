from ml_models.timeseries_model import TimeSeriesProbaModel
from sklearn import svm
import pickle


class SupportVectorMachineModel(TimeSeriesProbaModel):

    def __init__(self):
        super().__init__()
        self.name = "Support Vector Machine"
        self.algorithm = "SupportVectorMachine"


        self.hyperparameters = {
            "C": 1.0,
            "kernel": 'linear',
            "degree": 3,
            "gamma": 'scale',
            "coef0": 0.0,
            "shrinking": True,
            "probability": True,
            "tol": 0.001,
            "cache_size": 1000,
            "class_weight": None,
            "verbose": False,
            "max_iter": -1,
            "decision_function_shape": 'ovr',
            "break_ties": False,
            "random_state": 42
        }

        self.model = self._init()

    def _init(self):
        svm_model = svm.SVC()
        svm_model.set_params(**self.hyperparameters)
        return svm_model

    def train_model(self, training_data):
        """Function that starts the learning process of the SVM and stores the resulting model after completion"""

        # Init forest and read training data
        label = training_data["ards"]
        predictors = training_data.loc[:, training_data.columns != 'ards']

        # Learn and store resulting model
        self.model = self.model.fit(predictors, label)
        self.trained = True

    def predict(self, data):
        return self.model.predict(data)

    def predict_proba(self, data):
        return self.model.predict_proba(data)

    def get_params(self):
        return self.model.get_params(deep=True)

    def set_params(self, params: dict):
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
