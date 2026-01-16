
from ml_models.timeseries_model import TimeSeriesProbaModel
from sklearn.ensemble import AdaBoostClassifier
import pickle
import logging
import numpy as np

logger = logging.getLogger(__name__)

class AdaBoostModel(TimeSeriesProbaModel):

    def __init__(self):
        super().__init__()

        self.name = "AdaBoost"
        self.algorithm = "AdaBoost"
        self.hyperparameters = {
            "estimator": None,
            "n_estimators": 50,
            "learning_rate": 1.0,
            "algorithm": 'SAMME',
            "random_state": 42
        }
        self.model = self._init()

    def _init(self):
        adaboost = AdaBoostClassifier()
        adaboost.set_params(**self.hyperparameters)
        return adaboost

    def train_model(self, training_data):

        y = training_data["ards"]
        logger.info(f"ARDS data unique values: {y.unique()}")
        logger.info(f"ARDS data value counts: {y.value_counts()}")
        logger.info(f"ARDS data dtype: {y.dtype}")
        logger.info(f"ARDS data has NaN: {y.isna().any()}")
        logger.info(f"Number of NaN values: {y.isna().sum()}")
        logger.info(f"ARDS data: {y.describe()}")
        
        # Convert to standard numpy array to avoid pandas dtype issues
        y = y.astype(int).values
        logger.info(f"After conversion - y shape: {y.shape}, y dtype: {y.dtype}, unique values: {np.unique(y)}")
        
        X = training_data.loc[:, training_data.columns != 'ards']
        logger.info(f"Rest of training data: {X.describe()}")
        self.model.fit(X, y)
        self.trained = True

    def predict(self, patient_data):
        return self.model.predict(patient_data)

    def predict_proba(self, data):
        return self.model.predict_proba(data)

    def has_predict_proba(self):
        return True

    def get_params(self):
        return self.model.get_params()

    def set_params(self, params: dict[str, any]):
        for key, value in params.items():
            if key in self.hyperparameters:
                self.hyperparameters[key] = value
        self.model.set_params(**self.hyperparameters)

    def save_model(self, filepath):
        with open(filepath + f"{self.algorithm}_{self.name}.pkl", 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, filepath):
        with open(filepath + f"{self.algorithm}_{self.name}.pkl", 'rb') as f:
            self.model = pickle.load(f)
