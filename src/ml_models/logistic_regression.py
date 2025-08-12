from ml_models.timeseries_model import TimeSeriesProbaModel

from sklearn.linear_model import LogisticRegression
import pickle
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class LogisticRegressionModel(TimeSeriesProbaModel):

    def __init__(self):
        super().__init__()
        self.name = "Logistic_regression"
        self.algorithm = "LogisticRegression"
        self.hyperparameters = {
            "penalty": "l2",
            "dual": False,
            "tol": 0.0001,
            "C": 1.0,
            "fit_intercept": True,
            "intercept_scaling": 1,
            "class_weight": None,
            "random_state": 42,
            "solver": "saga",
            "max_iter": 100000,
            "multi_class": "auto",
            "verbose": 0,
            "warm_start": False,
            "n_jobs": -1,
            "l1_ratio": None,
        }

        self.penalty = 'l2'
        self.solver = 'saga'
        self.tol = 0.0001
        self.C = 1.0
        self.fit_intercept = True
        self.intercept_scaling = 1
        self.class_weight = None
        self.random_state = 42
        self.max_iter = 100000
        self.multi_class = 'auto'
        self.verbose = 0
        self.warm_start = False
        self.n_jobs = -1
        self.l1_ratio = None
        self.model = self._init_lr()

    def train_model(self, training_data):
        label = training_data["ards"]
        predictors = training_data.loc[:, training_data.columns != 'ards']

        # Modell trainieren
        self.model = self.model.fit(predictors, label)
        self.calculate_vif(predictors)
        self.trained = True

    def predict(self, data):
        return self.model.predict(data)

    def predict_proba(self, data):
        return self.model.predict_proba(data)

    def set_params(self, params):
        for key, value in params.items():
            if key in self.hyperparameters:
                self.hyperparameters[key] = value
        self.model.set_params(**self.hyperparameters)

    def get_params(self):
        return self.model.get_params(deep=True)

    def has_predict_proba(self):
        return True

    #Die Methode muss ein Array zurückgeben, welches für jede Zeile des Dataframes ein Array mit zwei Werten enthält. Der erste Wert gibt die Wahrscheinlichkeit an, dass es sich bei diesem Datensatz nicht um ARDS handelt, und der zweite gibt die Wahrscheinlichkeit an, dass es sich um ARDS handelt.

    def calculate_vif(self, data):
        logger.info("Calculating VIF...")
        vif_data = pd.DataFrame()
        #create empty pandas dataframe
        #vif_data["Feature"] = data.predictors
        vif_data["Feature"] = data.columns
        #assign columns
        vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
        #For each column index i, the VIF is calculated using the variance_inflation_factor function from the statsmodels.stats.outliers_influence module, applied to the entire input data (data.values) and the index i.
        vif_data.round(1)
        logger.info(vif_data)

    def _init_lr(self) -> LogisticRegression:
        """Function that intializes the Random Forest"""

        # Init LR
        logistic_regression = LogisticRegression()
        logistic_regression.set_params(**self.hyperparameters)
        return logistic_regression

    def save_model(self, filepath):
        file = open(filepath + f"{self.algorithm}_{self.name}.pkl", "wb")
        pickle.dump(self.model, file)

    def load_model(self, filepath):
        file = open(filepath + f"{self.algorithm}_{self.name}.pkl", "rb")
        self.model = pickle.load(file)


