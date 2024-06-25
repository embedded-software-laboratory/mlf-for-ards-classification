from models.model_interface import Model
from sklearn import svm
import pickle

class Support_vector_machine(Model):

    def __init__(self):
        super().__init__()
        self.name = "Support Vector Machine"
        self.model = svm.SVC(kernel="linear", probability=True)

    def train_model(self, training_data):
        """Function that starts the learning process of the SVM and stores the resulting model after completion"""
        
        # Init forest and read training data
        label = training_data["ards"]
        predictors = training_data.loc[:, training_data.columns != 'ards']

        # Learn and store resulting model
        self.model = self.model.fit(predictors, label)

    def predict(self, data): 
        return self.model.predict(data)
    
    def predict_proba(self, data):
        return self.model.predict_proba(data)
    
    def save(self, filepath):
        file = open(filepath + ".txt", "wb")
        pickle.dump(self.model, file)
    
    def load(self, filepath):
        file = open(filepath + ".txt", "rb")
        self.model = pickle.load(file)

    def has_predict_proba(self):
        return True
