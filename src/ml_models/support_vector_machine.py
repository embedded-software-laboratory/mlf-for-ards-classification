from ml_models.model_interface import Model
from ml_models.timeseries_model import TimeSeriesModel
from sklearn import svm
import pickle


class SupportVectorMachineModel(TimeSeriesModel):

    def __init__(self):
        super().__init__()
        self.name = "Support Vector Machine"
        self.algorithm = "Support Vector Machine"

        self.model = svm.SVC(kernel="linear", probability=True)

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

    def save_model(self, filepath):
        file = open(filepath + ".txt", "wb")
        pickle.dump(self.model, file)

    def load_model(self, filepath):
        file = open(filepath + ".txt", "rb")
        self.model = pickle.load(file)

    def has_predict_proba(self):
        return True
