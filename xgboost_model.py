import model_interface
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import sklearn

class XGBoost(model_interface.Model):

    def __init__(self):
        self.name = "XGBoost"
        self.model = XGBClassifier(eval_metric='mlogloss')
        self.le = LabelEncoder()
        self.le.fit_transform([0, 1])

    def train_model(self, training_data):
        label = training_data["ards"]
        predictors = training_data.loc[:, training_data.columns != 'ards']

        y_train = self.le.fit_transform(label)
        self.model = self.model.fit(predictors, y_train)

    def predict(self, patient_data): 
        pred = self.model.predict(patient_data)
        pred = self.le.inverse_transform(pred)
        return pred
    
    def predict_proba(self, data):
        return self.model.predict_proba(data)
    
    def save(self, filepath):
        self.model.save_model(filepath + ".ubj")

    def load(self, filepath):
        self.model.load_model(filepath + ".ubj")