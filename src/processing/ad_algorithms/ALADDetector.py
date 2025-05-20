import logging
import os
import pickle
import sys
from typing import Any

import joblib
import pandas as pd
from pyod.models.alad import ALAD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from processing.ad_algorithms.BaseAnomalyDetector import BaseAnomalyDetector
from processing.ad_algorithms.torch_utils import check_directory


logger = logging.getLogger(__name__)

class ALADDetector(BaseAnomalyDetector):

    def __init__(self, **kwargs):
        super().__init__()
        self.sk_seed = int(kwargs.get("sk_seed", 42))
        self.name = "ALAD"
        self.algorithm = "ALAD"
        self.run_dir = str(kwargs.get("run_dir", "../Data/Models/AnomalyDetection/ALAD"))
        self.checkpoint_dir = str(kwargs.get("checkpoint_dir", "../Data/Models/AnomalyDetection/ALAD"))
        self.prepared_data_dir = str(
            kwargs.get("prepared_data_dir", "/work/rwth1474/Data/AnomalyDetection/prepared_data/ALAD"))
        self.anomaly_data_dir = self.anomaly_data_dir = str(
            kwargs.get("anomaly_data_dir",
                       f"/work/rwth1474/Data/AnomalyDetection/anomaly_data/ALAD/{self.name}"))
        check_directory(str(self.run_dir))
        check_directory(str(self.checkpoint_dir))
        check_directory(str(self.prepared_data_dir))
        check_directory(str(self.anomaly_data_dir))
        self.hyperparameters = dict(kwargs.get("hyperparameters", {}))
        self._datasets_to_create = list(kwargs.get('dataset_to_create', []))
        self.model = {}
        self.needs_full_data = True
        self.trainable = True
        self.train_percentage = float(kwargs.get('train_percentage', 0.1))
        self.test_percentage = 1- self.train_percentage
        self.load_data = bool(kwargs.get('load_data', True))
        self.save_data = bool(kwargs.get('save_data', True))
        self.retrain_models = dict(kwargs.get('retrain_models', {}))
        self._update_retrain_models()

    def _update_retrain_models(self):
        for dataset in self._datasets_to_create:
            if dataset not in self.retrain_models:
                self.retrain_models[dataset] = True
            else:
                self.retrain_models[dataset] = False

    def create_meta_data(self):
        pass

    def _load_prepared_data(self, storage_info: str, type_of_dataset: str) -> Any:


        if type_of_dataset == "train":
            dataset = pd.read_pickle(storage_info)
            return dataset, [], pd.DataFrame()
        elif type_of_dataset == "predict":
            dataset = pd.read_pickle(storage_info)


            relevant_path = storage_info.replace("features", "relevant")
            relevant = pd.read_pickle(relevant_path)
            patients_to_remove_path = storage_info.replace("features", "patients_to_remove")
            with open(patients_to_remove_path, "rb") as f:
                patients_to_remove = pickle.load(f)
            return dataset, patients_to_remove, relevant
        else:
            logger.error("Unknown dataset type. Exiting...")
            sys.exit(1)



    def _train_ad_model(self, data_training, data_validation, **kwargs):

        raise NotImplementedError()

    def _predict(self, dataframe: pd.DataFrame, **kwargs) -> dict:
        raise NotImplementedError()


    def _prepare_data(self, dataframe: pd.DataFrame, save_data: bool = False, overwrite: bool = False) -> dict:
        diagnosis =  dataframe[["patient_id", "ards"]].drop_na(subset=["ards"]).reset_index(drop=True)
        train_patients, test_patients = train_test_split(diagnosis, test_size=self.test_percentage, random_state=self.sk_seed, shuffle=True,
                                                         stratify=diagnosis["ards"])
        train_patients_ids = train_patients["patient_id"].unique()
        test_patients_ids = test_patients["patient_id"].unique()
        train_data = dataframe[dataframe["patient_id"].isin(train_patients_ids)].reset_index(drop=True)
        test_data = dataframe[dataframe["patient_id"].isin(test_patients_ids)].reset_index(drop=True)
        data_dict = {
            "train_data": train_data,
            "val_data": None,
            "test_data": test_data,

        }
        if not self._datasets_to_create:
            self._datasets_to_create = [{"name": column,
                                         "features": column}
                                        for column in dataframe.columns if column not in self.columns_to_check]
        if "prepare" in self.active_stages:
            datatypes_to_prepare = ["test", "train"]
        elif "train" in self.active_stages:
            datatypes_to_prepare = ["train"]
        elif "predict" in self.active_stages:
            datatypes_to_prepare = ["test"]
        else:
            return {}

        for dataset_to_create in self._datasets_to_create:

            self._prepare_data_multi(datatypes_to_prepare, data_dict, dataset_to_create , overwrite)
        return data_dict

    @staticmethod
    def _get_filename_from_dataset_config(dataset_to_create: dict, type_of_dataset: str) -> str:
        name_str = dataset_to_create["name"]
        features_str = "+".join(dataset_to_create["features"])

        filename_str = f"{name_str}_{features_str}_{type_of_dataset}"
        return filename_str

    def _prepare_data_multi(self, datatypes_to_prepare, data_dict: dict, item,  overwrite: bool = False) -> None:

        for dataset_type in datatypes_to_prepare:
            self._prepare_dataset(data_dict,  item, dataset_type, overwrite)

    def _prepare_dataset(self, data_dict: dict, item: dict, dataset_type: str, overwrite: bool) -> None:
        feature_file = os.path.join(self.prepared_data_dir, f"{item['name']}_{dataset_type}_features.pkl")

        if not os.path.exists(feature_file)  or overwrite:
            self._prepare_data_step(data_dict[dataset_type], item, True, dataset_type)

    def _prepare_data_step(self, dataframe: pd.DataFrame, dataset_to_create: dict, save_data, type_of_dataset: str) -> (pd.DataFrame, list, pd.DataFrame):
        name = dataset_to_create["name"]
        relevant_columns = dataset_to_create["features"]
        relevant_data = dataframe[relevant_columns + ["patient_id", "time"]]
        relevant_data = relevant_data.dropna(subset=relevant_columns, how="any", axis=0).reset_index(drop=True)
        logger.info(f"Preparing {type_of_dataset} data for {name}...")
        relevant_patients = relevant_data["patient_id"].unique().tolist()

        if len(relevant_data.index) == 0:
            logger.info(f"Not enough data for {type_of_dataset}, skipping {name}...")
            return None, [], pd.DataFrame()
        to_scale = relevant_data[relevant_columns].copy(deep=True)
        if type_of_dataset == "train":
            scaler = MinMaxScaler()
            scaler.fit(to_scale)
            scaled = scaler.transform(to_scale)
            joblib.dump(scaler, os.path.join(self.prepared_data_dir, f"{name}_scaler.pkl"))
        else:
            try:
                scaler = joblib.load(os.path.join(self.prepared_data_dir, f"{name}_scaler.pkl"))
                scaled = scaler.transform(to_scale)
            except FileNotFoundError:
                logger.error(f"No scaler found for {name}. Create a new dataset. Exiting...")
                sys.exit(1)
        dataset = pd.DataFrame(scaled, columns=relevant_columns, index=to_scale.index)

        if dataset.empty:
            logger.info(f"No data found for {name}. Skipping...")
            return None, [], pd.DataFrame()

        if save_data:
            dataset_file_name = self._get_filename_from_dataset_config(dataset_to_create, type_of_dataset)
            feature_path = os.path.join(self.prepared_data_dir, f"{dataset_file_name}_features.pkl")
            contained_patients_path = os.path.join(self.prepared_data_dir, f"{dataset_file_name}_patients.pkl")
            self._save_file(dataset, feature_path, True)
            self._save_file(relevant_patients, contained_patients_path, True)

            if dataset_to_create == "test":
                relevant_path = os.path.join(self.prepared_data_dir, f"{dataset_file_name}_relevant.pkl")
                patients_to_remove_path = os.path.join(self.prepared_data_dir, f"{dataset_file_name}_patients_to_remove.pkl")
                patients_to_remove = list(set(dataframe["patient_id"].unique().tolist()) - set(relevant_patients))
                self._save_file(patients_to_remove, patients_to_remove_path, True)
                self._save_file(relevant_data, relevant_path, True)
        if type_of_dataset != "test":
            return dataset, [], pd.DataFrame()
        else:
            return dataset, patients_to_remove, relevant_data













