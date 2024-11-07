#(informal) interface for all ards detection ml_models
class Model:
    def __init__(self):
        self.model = None
        self.name = None
        self.type = None
        self.algorithm = None
        self.trained = False
        self._storage_location = None
        self.training_evaluation = None

    def train_model(self, training_data):
        raise NotImplementedError

    def predict(self, patient_data):
        raise NotImplementedError

    def predict_proba(self, data):
        raise NotImplementedError

    def has_predict_proba(self):
        raise NotImplementedError

    def save_model(self, filepath):
        raise NotImplementedError

    def load_model(self, filepath):
        raise NotImplementedError

    @property
    def storage_location(self):
        raise NotImplementedError

    @storage_location.setter
    def storage_location(self, location):
        raise NotImplementedError
