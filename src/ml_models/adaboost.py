
from ml_models.timeseries_model import TimeSeriesModel
from sklearn.ensemble import AdaBoostClassifier
import pickle


class AdaBoostModel(TimeSeriesModel):

    def __init__(self):
        super().__init__()

        self.name = "AdaBoost"
        self.algorithm = "AdaBoost"
        self.model = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=0)

    def train_model(self, training_data):

        y = training_data["ards"]
        X = training_data.loc[:, training_data.columns != 'ards']
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

    def save_model(self, filepath):
        with open(filepath + ".pkl", 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, filepath):
        with open(filepath + ".pkl", 'rb') as f:
            self.model = pickle.load(f)
