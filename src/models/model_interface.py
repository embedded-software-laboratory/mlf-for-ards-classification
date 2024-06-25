
#(informal) interface for all ards detection models
class Model:
    def __init__(self):
        self.model = None
        self.name = None

    def train_model(self, training_data):
        raise NotImplementedError

    def predict(self, patient_data):
        raise NotImplementedError

    def predict_proba(self, data):
        raise NotImplementedError

    def has_predict_proba(self):
        raise NotImplementedError

    def save(self, filepath):
        raise NotImplementedError

    def load(self, filepath):
        raise NotImplementedError
    