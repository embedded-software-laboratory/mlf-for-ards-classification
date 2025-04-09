import copy
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


from tensorflow import keras

from processing.ad_algorithms.AnomalyDetector import AnomalyDetector
from processing.ad_algorithms.WindowGenerator import WindowGenerator
from processing.datasets_metadata import AnomalyDetectionMetaData
tf.config.threading.set_inter_op_parallelism_threads(8)
class DeepAntPredictor:

    def __init__(self, output_dim: int=1, hidden_units: int = 256, max_epochs: int =20, name: str = "DeepAntPredictor"):
        self.output_dim = output_dim
        self.name = name
        self.hidden_units = hidden_units
        self.max_epochs = max_epochs
        self.model = None
        self._build_model()
        self.trained = False



    def _build_model(self):
        self.model = keras.Sequential([
            keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dense(units=self.hidden_units, activation='relu'),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(units=self.output_dim, activation="relu")
        ])

    def compile_and_fit(self, data_training, data_validation, patience=5):
        early_stopping_criteria = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min',
            verbose=1
        )
        checkpoint = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        self.model.compile(optimizer=keras.optimizers.Adam() , loss=keras.losses.MSE, metrics=[keras.metrics.MAE])
        history = self.model.fit(data_training, epochs=self.max_epochs, validation_data=data_validation, callbacks=[early_stopping_criteria, checkpoint])
        self.trained = True
        return history

    def predict(self, sequence):
        return self.model.predict(sequence)


class DeepAntAnomalyDetector(AnomalyDetector):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = {}
        self.deep_ant_output_dim = int(kwargs.get('output_dim', 1))
        self.deep_ant_hidden_units = int(kwargs.get('hidden_units', 256))
        self.deep_ant_max_epochs = int(kwargs.get('max_epochs', 20))
        self.datasets_to_create = list(kwargs.get('dataset_to_create', []))
        self.std_rate = float(kwargs.get('std_rate', 2))

        self.val_percentage = float(kwargs.get('val_percentage', 0.1))
        self.train_percentage = float(kwargs.get('train_percentage', 0.1))
        self.test_percentage = float(kwargs.get('test_percentage', 0.1))
        self.seed = int(kwargs.get('seed', 42))
        self.needs_full_data = True
        self.window_generator_config = dict(kwargs.get('window_generator_config', {}))

    def create_meta_data(self):
        meta_data_dict = super().create_meta_data()
        meta_data_dict["anomaly_detection_algorithm"] = self.type
        meta_data_dict["algorithm_specific_settings"] = {"length_prediction_horizon": self.deep_ant_output_dim,
                                                         "hidden_units": self.deep_ant_hidden_units,
                                                         "max_epochs": self.deep_ant_max_epochs,
                                                         "val_percentage": self.val_percentage,
                                                         "train_percentage": self.train_percentage,
                                                         "test_percentage": self.test_percentage,
                                                         "seed": self.seed,
                                                         "window_generator_config": self.window_generator_config,}

        return AnomalyDetectionMetaData(**meta_data_dict)

    def run(self,  dataframe_detection: pd.DataFrame, job_count: int, total_jobs: int) -> pd.DataFrame:

        prepared_data = self._prepare_data(dataframe_detection)
        results = []
        existing_labels = []
        for item in self.datasets_to_create:
            labels = item["labels"]
            for label in labels:
                if label not in existing_labels:
                    existing_labels.append(label)
                else:
                    raise ValueError("Double label detected for label: " + label)
        for item in self.datasets_to_create:
            print("Running dataset: ", item["name"])
            results.append(self._run_step(prepared_data, item))
        df_anomaly = pd.DataFrame()
        for item in results:
            df_anomaly.merge(item, how="outer", left_index=True, right_index=True)
        anomaly_columns = df_anomaly.columns.tolist()
        rename_dict = {column: column + "_anomaly" for column in anomaly_columns}
        remaining_data = prepared_data["test"]
        df_anomaly = df_anomaly.rename(columns=rename_dict)
        relevant_data = prepared_data["test"][df_anomaly.columns.tolist()]
        df_anomaly = df_anomaly.fillna(False)
        fixed_df = self._handle_anomalies({"results": df_anomaly},  relevant_data, remaining_data)
        return fixed_df






    def _prepare_data(self, dataframe: pd.DataFrame) -> dict[str, pd.DataFrame]:
        patient_ids = list(dataframe["patient_id"].unique())
        np_patients = np.array(patient_ids)
        train_patients, remaining_patients = train_test_split(np_patients, test_size=1 - self.train_percentage,
                                                              random_state=self.seed, shuffle=True)
        val_patients, test_patients = train_test_split(remaining_patients, test_size=self.test_percentage / (
                    self.test_percentage + self.val_percentage), random_state=self.seed, shuffle=True)
        train_data = dataframe[dataframe["patient_id"].isin(train_patients)].reset_index()
        val_data = dataframe[dataframe["patient_id"].isin(val_patients)].reset_index()
        test_data = dataframe[dataframe["patient_id"].isin(test_patients)].reset_index()
        result_dict = {"train": train_data, "val": val_data, "test": test_data}
        if not self.datasets_to_create:
            self.datasets_to_create = [{"name": column, "labels": [column], "features": [column]} for column in
                                       dataframe.columns if column not in self.columns_not_to_check]
        return result_dict



    def _run_step(self, data: dict[str, pd.DataFrame], dataset_to_create: dict[str, Union[list, str]]) -> pd.DataFrame:

        name = dataset_to_create["name"]
        train = data["train"]
        val = data["val"]
        test = data["test"]
        test_patients = test["patient_id"].unique().tolist()
        val_patients = val["patient_id"].unique().tolist()
        train_patients = train["patient_id"].unique().tolist()
        relevant_columns = dataset_to_create["labels"] + dataset_to_create["features"] + ["patient_id", "time"]
        relevant_train = train[relevant_columns]
        relevant_train = relevant_train.dropna(how='any', axis=0).reset_index(drop=True)
        relevant_val = val[relevant_columns]
        relevant_val = relevant_val.dropna(how='any', axis=0).reset_index(drop=True)
        relevant_test = test[relevant_columns]
        relevant_test = relevant_test.dropna(how='any', axis=0).reset_index(drop=True)
        patient_divisions_train = self._build_patient_dict(relevant_train)
        patient_divisions_val = self._build_patient_dict(relevant_val)
        patient_divisions_test = self._build_patient_dict(relevant_test)

        scaler = MinMaxScaler()
        scaler.fit(relevant_train)
        scaled_train = scaler.transform(relevant_train)
        scaled_val = scaler.transform(relevant_val)
        scaled_test = scaler.transform(relevant_test)
        relevant_train = pd.DataFrame(scaled_train, columns=relevant_train.columns, index=relevant_train.index)
        relevant_val = pd.DataFrame(scaled_val, columns=relevant_val.columns, index=relevant_val.index)
        relevant_test = pd.DataFrame(scaled_test, columns=relevant_test.columns, index=relevant_test.index)
        relevant_train.drop(columns="patient_id", inplace=True)
        relevant_val.drop(columns="patient_id", inplace=True)
        relevant_test.drop(columns="patient_id", inplace=True)
        train_dataset_list = []
        val_dataset_list = []
        test_dataset_list = []
        for train_patients in patient_divisions_train.values():
            patient_train = relevant_train[train_patients[0]:train_patients[1]]
            windowed_data = WindowGenerator(patient_train, **self.window_generator_config).generate_dataset()
            train_dataset_list.append(windowed_data)

        for val_patients in patient_divisions_val.values():
            patient_val = relevant_val[val_patients[0]:val_patients[1]]
            windowed_data = WindowGenerator(patient_val, **self.window_generator_config).generate_dataset()
            val_dataset_list.append(windowed_data)



        train_dataset = None
        for i in range(len(train_dataset_list)):
            print(f"{i}/{len(train_dataset_list)}")
            if train_dataset is None and train_dataset_list[i] is not None:
                train_dataset = train_dataset_list[i]
            elif train_dataset_list[i] is not None:
                train_dataset = train_dataset.concatenate(train_dataset_list[i])


        val_dataset = None
        for i in range(len(val_dataset_list)):
            print(f"{i}/{len(val_dataset_list)}")
            if val_dataset is None and val_dataset_list[i] is not None:
                val_dataset = val_dataset_list[i]
            elif val_dataset_list[i] is not None:
                val_dataset = val_dataset.concatenate(val_dataset_list[i])



        model = DeepAntPredictor(output_dim=len(dataset_to_create["labels"]), hidden_units=self.deep_ant_hidden_units, max_epochs=self.deep_ant_max_epochs, name=name)
        model, history = self._train_ad_model(train_dataset, val_dataset, model=model)
        for test_patients in patient_divisions_test.values():
            patient_test = relevant_test[test_patients[0]:test_patients[1]]
            windowed_data = WindowGenerator(patient_test, **self.window_generator_config).generate_dataset()
            test_dataset_list.append(windowed_data)

        test_dataset = None
        for i in range(len(test_dataset_list)):
            if test_dataset is None and test_dataset_list[i] is not None:
                test_dataset = test_dataset_list[i]
                print(len(list(test_dataset.unbatch())))
            elif test_dataset_list[i] is not None:
                test_dataset = test_dataset.concatenate(test_dataset_list[i])

        print(len(list(test_dataset.unbatch())))
        anomalies = self._predict(test_dataset, model=model)
        self.model[name] = model
        for column in dataset_to_create["labels"]:
            test[column + "_anomaly"] = anomalies["results"]
        #if self.handling_strategy == "use_prediction":
        #    pass
        #    for column in dataset_to_create["labels"]:
        #
        #        test[column + "_prediction"] = anomalies["predictions"]
        columns_to_return = [column + "_anomaly" for column in dataset_to_create["labels"]]
        if self.handling_strategy == "use_prediction":
            columns_to_return += [column + "_prediction" for column in dataset_to_create["labels"]]

        return test[columns_to_return]

    @staticmethod
    def _build_patient_dict(dataframe: pd.DataFrame) -> dict:
        patient_dict = {}
        patient_ids = dataframe["patient_id"].unique().tolist()
        patient_max_time_df = dataframe.groupby("patient_id")["time"].idxmax()
        patient_min_time_df = dataframe.groupby("patient_id")["time"].idxmin()
        for i in range(len(patient_ids)):
            patient_dict[patient_ids[i]] = (int(patient_min_time_df.iloc[i]),
                                            int(patient_max_time_df.iloc[i])
                                             )
        return patient_dict










    def _train_ad_model(self, data_training, data_validation, **kwargs):
        model = kwargs['model']
        if not model.trained:
            history = model.compile_and_fit(data_training, data_validation)
            return model, history
        else:
            print("Model already trained")
            return None





    def _predict(self, data, **kwargs) -> dict:
        model = kwargs['model']
        anomaly_dict = {"results": []}
        print(len(list(data.unbatch())))
        if not model.trained:
            raise ValueError("Model must be trained before prediction")
        else:
            predictions = model.predict(data)
            ground_truth = np.concatenate([y for x,y in data], axis=0).squeeze()
            anomaly_scores = np.linalg.norm(predictions - ground_truth, axis=1)
            threshold = self._calculate_threshold(anomaly_scores)
            anomalies = [True if score > threshold else False for score in anomaly_scores]
            anomaly_dict["results"] = anomalies
            # TODO: check how to incorporate predictions regarding data type etc
            if self.handling_strategy == "use_prediction":
                anomaly_dict["predictions"] = predictions
        return anomaly_dict





    @staticmethod
    def _calculate_threshold(anomaly_scores, std_rate=2):
        return np.mean(anomaly_scores) + std_rate * np.std(anomaly_scores)

    def _predict_proba(self):
        raise NotImplementedError()












    def _handle_anomalies(self, anomalies: dict, anomalous_data : pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        anomaly_df = anomalies["results"]
        if self.handling_strategy == "delete_value":
            fixed_df = self._delete_value(anomaly_df, original_data)
        elif self.handling_strategy == "delete_than_impute":
            fixed_df = self._delete_than_impute(anomaly_df, anomalous_data)
        elif self.handling_strategy == "delete_row_if_any_anomaly":
            fixed_df = self._delete_row_if_any_anomaly(anomaly_df, anomalous_data)
        elif self.handling_strategy == "delete_row_if_many_anomalies":
            fixed_df = self._delete_row_if_many_anomalies(anomaly_df, anomalous_data)
        elif self.handling_strategy == "use_prediction":
            raise ValueError("Fixing strategy 'use_prediction' is not implemented for PhysicalLimitsDetector")
        else:
            raise ValueError("Unknown fixing strategy")
        finished_df = original_data
        finished_df.update(fixed_df)
        return finished_df


