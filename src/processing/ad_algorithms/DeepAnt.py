import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras

from processing.ad_algorithms.AnomalyDetector import AnomalyDetector

class DeepAnt:

    def __init__(self, output_dim: int=1, hidden_units: int = 256, max_epochs: int =20):
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.max_epochs = max_epochs
        self.model = None
        self.trained = False


    def _build_model(self):
        self.model = keras.Sequential([
            keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(units=self.hidden_units, activation=None)
        ])

    def compile_and_fit(self, data_training, data_validation, patience=5):
        early_stopping_criteria = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min',
            verbose=1
        )
        checkpoint = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        self.model.compile(optimizer=keras.optimizers.Adam, loss=keras.losses.MSE, metrics=[keras.metrics.MAE])
        history = self.model.fit(data_training, epochs=self.max_epochs, validation_data=data_validation, callbacks=[early_stopping_criteria, checkpoint])
        self.trained = True
        return history

    def predict(self, sequence):
        return self.model.predict(sequence)


class DeepAntAnomalyDetector(AnomalyDetector):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        deep_ant_output_dim = int(kwargs.get('output_dim', 1))
        deep_ant_hidden_units = int(kwargs.get('hidden_units', 256))
        deep_ant_max_epochs = int(kwargs.get('max_epochs', 20))
        self.datasets_to_create = list(kwargs.get('dataset_to_create', []))
        self.std_rate = float(kwargs.get('std_rate', 2))
        self.model = DeepAnt(deep_ant_output_dim, deep_ant_hidden_units, deep_ant_max_epochs)
        self.val_percentage = float(kwargs.get('val_percentage', 0.1))
        self.train_percentage = float(kwargs.get('train_percentage', 0.1))
        self.test_percentage = float(kwargs.get('test_percentage', 0.1))
        self.seed = int(kwargs.get('seed', 42))




    def run(self,  dataframe_detection: pd.DataFrame, job_count: int, total_jobs: int) -> pd.DataFrame:

        identifiers = dataframe_detection['patient_id'].unique().tolist()
        results = []
        for id in identifiers:
            relevant_data = dataframe_detection[dataframe_detection['patient_id'] == id]
            results.append(self._run_patient(relevant_data))

        fixed_df = pd.concat(results).reset_index(drop=True)
        return fixed_df


    def _run_patient(self, df_patients_data: pd.DataFrame):
        prepared_dict = self._prepare_data(df_patients_data)
        dataset_training = prepared_dict['training']
        dataset_validation = prepared_dict['validation']
        dataset_detection = prepared_dict['detection']
        history = self._train_ad_model(dataset_training, dataset_validation)






    def _train_ad_model(self, data_training, data_validation):
        if not self.model.trained:
            return self.model.compile_and_fit(data_training, data_validation)
        else:
            print("Model already trained")
            return None





    def _predict(self, data) -> dict:
        if not self.model.trained:
            raise ValueError("Model must be trained before prediction")
        else:
            predictions = self.model.predict(data)
            ground_truth = np.concatenate([y for x,y in data], axis=0).squeeze()
            anomaly_scores = np.linalg.norm(predictions - ground_truth, axis=1)
            threshold = self._calculate_threshold(anomaly_scores)
            anomalies = [True if score > threshold else False for score in anomaly_scores]



    def _calculate_threshold(self, anomaly_scores, std_rate=2):
        return np.mean(anomaly_scores) + std_rate * np.std(anomaly_scores)

    def _predict_proba(self):
        raise NotImplementedError()

    def _prepare_data_dataset(self, dataframe: pd.DataFrame, dataset_to_create: dict[str, list[str]]) -> dict:

        relevant_columns = dataset_to_create["labels"] + dataset_to_create["features"]
        relevant_data = dataframe[relevant_columns]
        relevant_data = relevant_data.dropna(how='any', axis=0)
        train_data, remaining = train_test_split(relevant_data, test_size=1-self.train_percentage, random_state=self.seed, shuffle=True)
        remaining_percentage = self.test_percentage / (1-self.train_percentage)
        val_data, test_data = train_test_split(remaining, test_size=remaining_percentage, random_state=self.seed, shuffle=True)
        scaler = MinMaxScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data.reset_index(drop=True))
        val_data = scaler.transform(val_data.reset_index(drop=True))
        test_data = scaler.transform(test_data.reset_index(drop=True))



        # TODO create sequence data
        # TODO create seperated sequences for each requested column / combinations of columns

        raise NotImplementedError()

    def _handle_anomalies(self, anomalies: dict, anomalous_data : pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()


