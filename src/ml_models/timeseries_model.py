from ml_models.model_interface import Model


class TimeSeriesModel(Model):

    def __init__(self):
        super().__init__()
        self._storage_location = ""

    @property
    def storage_location(self):
        return self._storage_location

    @storage_location.setter
    def storage_location(self, location):
        self._storage_location = location
