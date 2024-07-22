from ml_models.model_interface import Model
import keras
from keras.layers import LSTM, Dropout, Dense
import numpy as np


class Recurrent_neural_network(Model):

    def __init__(self):
        super().__init__()
        self.name = "Recurrent neural network"

    def train_model(self, training_data):
        """Function that starts the learning process of the RCN and stores the resulting model after completion"""

        #predictors, label = self.generate_data(training_data.loc[:, training_data.columns != 'ARDS'], training_data["ARDS"], len(training_data))
        label = training_data["ards"]
        predictors = training_data.loc[:, training_data.columns != 'ards']
        # Init lstm and read training data
        self.model = keras.Sequential()
        self.model.add(LSTM(100, input_shape = (predictors.shape[1], 1))) #input_shape_1: = Anzahl an Features pro Timestep, input_shape_2 = Anzahl vorhandener Steps pro Time-Series
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation="sigmoid"))
        self.model.compile(loss="binary_crossentropy"
              , metrics=[keras.metrics.binary_accuracy]
              , optimizer="adam")
        #lstm.summary()

        # Learn and store resulting model
        self.model.fit(predictors.values, label, batch_size=64, epochs=10, verbose=0)

    def predict(self, data): 
        prediction_normalized = self.predict_proba_normalized(data)
        prediction_classified = []
        for i in prediction_normalized:
            if i >= 0.5:
                prediction_classified.append(1)
            else:
                prediction_classified.append(0)
        return prediction_classified
    
    def predict_proba_normalized(self, data):
        prediction = self.model.predict(data.values, verbose=0)
        min_value = min(prediction)
        max_value = max(prediction)
        return list(map(lambda x: (x-min_value)/(max_value-min_value), prediction))
    
    def predict_proba(self, data):
        prediction = self.predict_proba_normalized(data)
        result = []
        for p in prediction:
            temp = []
            temp.append(1-p[0])
            temp.append(p[0])
            result.append(temp)
        return(np.array(result))
    
    def save(self, filepath):
        self.model.save(filepath + ".h5")
    
    def load(self, filepath):
        self.model = keras.models.load_model(filepath + ".h5")
    
    def generate_data(self, X, y, data_length, sequence_length = 406, step = 1):
        X_local = []
        y_local = []
        for start in range(0, data_length - sequence_length, step):
            end = start + sequence_length
            X_local.append(X[start:end])
            y_local.append(y[end-1])
        return np.array(X_local), np.array(y_local)

    def has_predict_proba(self):
        return False
