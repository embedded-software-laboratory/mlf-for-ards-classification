from ml_models.model_interface import Model
from ml_models.timeseries_model import TimeSeriesModel
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

class XGBoostModel(TimeSeriesModel):

    def __init__(self):
        super().__init__()
        self.name = "XGBoost"
        self.algorithm = "XGBoost"
        self.eval_metric = 'mlogloss'
        self.random_state = 42
        self.model = XGBClassifier(eval_metric=self.eval_metric, random_state=self.random_state)
        self.le = LabelEncoder()
        self.le.fit_transform([0, 1])

    def train_model(self, training_data):
        label = training_data["ards"]
        predictors = training_data.loc[:, training_data.columns != 'ards']

        y_train = self.le.fit_transform(label)
        self.model = self.model.fit(predictors, y_train)
        self.trained = True

    def predict(self, patient_data): 
        pred = self.model.predict(patient_data)
        pred = self.le.inverse_transform(pred)
        return pred
    
    def predict_proba(self, data):
        return self.model.predict_proba(data)

    def get_params(self):
        return {'eval_metric': self.eval_metric}

    def save_model(self, filepath):
        self.model.save_model(filepath + ".ubj")

    def load_model(self, filepath):
        self.model.load_model(filepath + ".ubj")

    def has_predict_proba(self):
        return True


