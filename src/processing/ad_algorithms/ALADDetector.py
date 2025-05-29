import logging
import os
import pickle
import sys
from typing import Any

import joblib
import pandas as pd
from pandas.core.interchange import column
from pyod.models.alad import ALAD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from processing.ad_algorithms.BaseAnomalyDetector import BaseAnomalyDetector
from processing.ad_algorithms.torch_utils import check_directory
from processing.datasets_metadata import AnomalyDetectionMetaData

logger = logging.getLogger(__name__)

class ALADDetector(BaseAnomalyDetector):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        
        self.trained_models = []

    def _update_retrain_models(self):
        for dataset in self._datasets_to_create:
            if dataset not in self.retrain_models:
                self.retrain_models[dataset] = True
            else:
                self.retrain_models[dataset] = False

    def _create_meta_data_preparation(self, test_data: pd.DataFrame) -> dict:
        contained_patients = test_data["patient_id"].unique().tolist()
        meta_data_dict = {
            "algorithm_specific_settings": {

                "train_percentage": self.train_percentage,

                "test_percentage": 1 - self.train_percentage,
                "seed": self.sk_seed
            },
            "datasets": self._datasets_to_create,
            "contained_patients": contained_patients,
        }
        return meta_data_dict

    def create_meta_data(self):
        meta_data_dict =  super().create_meta_data()
        meta_data_dict["anomaly_detection_algorithm"] = self.type
        meta_data_dict["algorithm_specific_settings"] = {
            "train_percentage": self.train_percentage,
            "test_percentage": self.test_percentage,
            "datasets": self._datasets_to_create,
            "config": self.hyperparameters,
            "checkpoint_dir": self.checkpoint_dir,
            "prepared_data_dir": self.prepared_data_dir
        }
        return AnomalyDetectionMetaData(**meta_data_dict)

    def _load_prepared_data(self, path: str, type_of_dataset: str) -> Any:
        """Returns none because data handling is done by the setup function"""
        return None


    def _load_data(self, storage_info: str, type_of_dataset: str) -> Any:

        path = self.prepared_data_dir + "/" + f"{storage_info}_features.pkl"
        status = -1
        dataset = pd.DataFrame()
        patients_to_remove = []
        relevant = pd.DataFrame()
        
        try:
            path = self.prepared_data_dir + "/" + f"{storage_info}_features.pkl"
            if type_of_dataset == "train":
                dataset = pd.read_pickle(path)
                status = 0

            elif type_of_dataset == "test":
                dataset = pd.read_pickle(path)

                relevant_path = self.prepared_data_dir + "/" + f"{storage_info}_relevant.pkl"
                
                relevant = pd.read_pickle(relevant_path)
                
                patients_to_remove_path =  self.prepared_data_dir + "/" + f"{storage_info}_patients_to_remove.pkl"

                with open(patients_to_remove_path, "rb") as f:
                    patients_to_remove = pickle.load(f)#
                status = 0

            else:
                logger.error(f"Unknown dataset type. {type_of_dataset}")
        except Exception as e:
            logger.error(f"Failed to load data for {storage_info} {type_of_dataset}: {e}")
            status = -1
            dataset = pd.DataFrame()
            relevant = pd.DataFrame()
            patients_to_remove = []
        return  status, dataset, patients_to_remove, relevant


    def _handle_data_step(self, dataset_to_create: dict, data: pd.DataFrame, stage, load_data: bool = True, save_data: bool=True) -> (int, pd.DataFrame, list, pd.DataFrame):
        name = dataset_to_create["name"]
        dataset = None
        status = -1
        patients_to_remove = []
        relevant_data = pd.DataFrame()
        if stage == "predict":
            stage = "test"
        if load_data:
            filename = self._get_filename_from_dataset_config(dataset_to_create, stage)
            status, dataset, patients_to_remove, relevant_data = self._load_data(filename, stage)
        if status != 0:
            status , dataset, patients_to_remove, relevant_data = self._prepare_data_step(data, dataset_to_create, save_data, stage)
        if  dataset is None or dataset.empty :
            logger.info(f"No dataset for stage {stage} for parameter {name}. Skipping.")
        return status, dataset, patients_to_remove, relevant_data


    def _setup_alad(self, dataset_to_create: dict, data: pd.DataFrame, stage: str, load_data: bool = True, save_data: bool = True) -> tuple:
        name = dataset_to_create["name"]
        dataset = pd.DataFrame()
        patients_to_remove = []
        relevant_data = pd.DataFrame()
        status, dataset, patients_to_remove, relevant_data = self._handle_data_step(dataset_to_create, data, stage, load_data, save_data)
        if stage == "train":
            if not self.hyperparameters:
                self.model[name] = ALAD()
            else:
                self.model[name] = ALAD(**self.hyperparameters)



        elif stage == "predict":
            if not name in self.model.keys() or not name in self.trained_models:

                model_location = os.path.join(self.checkpoint_dir, f"model_{name}.ckpt")
                model_file_exists = os.path.exists(model_location)

                if not model_file_exists:
                    status = -1
                    logger.info(f"No model file for parameter {name}. Please train a model before using prediction")
                else:
                    self.model[name] = joblib.load(model_location)
                    self.trained_models.append(name)
        if status != 0:
            logger.info(f"Problem while setting up model for stage {stage} for parameter {name}. Skipping...")


        return status, dataset, patients_to_remove, relevant_data





    def _build_datasets_from_dataframe_or_files(self, data: pd.DataFrame) -> None:
        if not self._datasets_to_create:
            if not data is None and not data.empty:
                self._datasets_to_create = [{"name": column,
                                             "features": column}
                                            for column in data.columns if column not in self.columns_to_check]
            else:
                contained_files = os.listdir(self.prepared_data_dir)
                contained_train = [file.removesuffix("_train_features.pkl") for file in contained_files if file.endswith("_train_features.pkl")]
                contained_test = [file.removesuffix("_test_features.pkl") for file in contained_files if file.endswith("_test_features.pkl")]
                all_present = list(set(contained_train).intersection(set(contained_test)))
                for filename in all_present:
                    self._datasets_to_create.append(self._get_dataset_config_from_filename(filename))


    def _train_ad_model(self, data_training, data_validation, **kwargs):
        self._build_datasets_from_dataframe_or_files(data_training)

        for dataset in self._datasets_to_create:
            name = dataset["name"]
            retrain_model = self.retrain_models.get(name, False)
            model_location = os.path.join(self.checkpoint_dir, f"model_{name}.ckpt")
            logger.info(f"Check if model for {name} exists in {model_location}")
            model_exists = os.path.exists(model_location)
            model_training = retrain_model or not model_exists
            if model_training:
                logger.info(f"Training model for {name}...")
                if name not in self.model.keys():
                    status, dataset_features, _, _ = self._setup_alad(dataset,  data_training, "train",  self.load_data, self.save_data)
                else:
                    status, dataset_features, _, _ = self._handle_data_step(dataset, data_training, "train", self.load_data, self.save_data)
                if status == 0:
                    self.model[name].fit(dataset_features)
                    self.trained_models.append(name)
                    joblib.dump(self.model[name], model_location)

                else:
                    logger.info(f"Problem while training model for {name}. Skipping.")
                    continue
            else:
                logger.info(f"Model already exists for {name}")






    def _predict(self, dataframe: pd.DataFrame, **kwargs) -> dict:
        logger.info(f"Start anomaly detection")
        self._build_datasets_from_dataframe_or_files(None)
        relevant_df_list = []
        anomaly_df_list = []
        for dataset in self._datasets_to_create:
            
            name = dataset["name"]
            logger.info(f"Predicting for parameter {name}...")
            status, dataset_features, patients_to_remove, relevant_data = self._setup_alad(dataset, dataframe, "predict", self.load_data, self.save_data)
            
            if status == 0:
                anomalies = self.model[name].predict(dataset_features)
            else:
                logger.info(f"Problem while predicting for {name}. Skipping.")
                continue
            anomaly_df = pd.DataFrame()
            anomaly_df["patient_id"] = relevant_data["patient_id"]
            anomaly_df["time"] = relevant_data["time"]
            anomaly_df[name] = [anomaly == 1 for anomaly in anomalies]
            anomaly_df_list.append(anomaly_df)
            relevant_df_list.append(relevant_data)
            logger.info(f"Predicting for parameter {name} finished...")
        anomaly_df = pd.DataFrame()
        relevant_df = pd.DataFrame()
        for i in range(len(anomaly_df_list)):
            if i == 0:
                anomaly_df = anomaly_df_list[i]
            else:
                anomaly_df = pd.merge(anomaly_df, anomaly_df_list[i], how="outer", on=["patient_id", "time"])
        for i in range(len(relevant_df_list)):
            if i == 0:
                relevant_df = relevant_df_list[i]
            else:
                relevant_df = pd.merge(relevant_df_list[i], relevant_df_list[i], how="outer", on=["patient_id", "time"])


        anomaly_df.fillna(False, inplace=True)
        self._save_anomaly_df(anomaly_df)
        anomaly_count_dict = self._calculate_anomaly_count(anomaly_df, relevant_df)
        logger.info(f"Finished anomaly detection")

        return {
            "anomaly_df": anomaly_df,
            "anomaly_count": anomaly_count_dict,
        }



    def _prepare_data(self, dataframe: pd.DataFrame, save_data: bool = False, overwrite: bool = False) -> dict:
        diagnosis =  dataframe[["patient_id", "ards"]].dropna(subset=["ards"]).reset_index(drop=True)
        train_patients, test_patients = train_test_split(diagnosis, test_size=self.test_percentage, random_state=self.sk_seed, shuffle=True,
                                                         stratify=diagnosis["ards"])
        train_patients_ids = train_patients["patient_id"].unique()
        test_patients_ids = test_patients["patient_id"].unique()
        train_data = dataframe[dataframe["patient_id"].isin(train_patients_ids)].reset_index(drop=True)
        test_data = dataframe[dataframe["patient_id"].isin(test_patients_ids)].reset_index(drop=True)
        data_dict = {
            "train": train_data,
            "val": None,
            "test": test_data

        }
        if not self._datasets_to_create:
            self._datasets_to_create = [{"name": column,
                                         "features": [column]}
                                        for column in dataframe.columns if column not in self.columns_not_to_check]
        if "prepare" in self.active_stages:
            datatypes_to_prepare = ["train", "test"]
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

    def _prepare_data_step(self, dataframe: pd.DataFrame, dataset_to_create: dict, save_data, type_of_dataset: str) -> (int, pd.DataFrame, list, pd.DataFrame):
        name = dataset_to_create["name"]
        relevant_columns = dataset_to_create["features"]
        
        relevant_data = dataframe[relevant_columns + ["patient_id", "time"]]
        relevant_data = relevant_data.dropna(subset=relevant_columns, how="any", axis=0).reset_index(drop=True)
        logger.info(f"Preparing {type_of_dataset} data for {name}...")
        relevant_patients = relevant_data["patient_id"].unique().tolist()
        patients_to_remove = list(set(dataframe["patient_id"].unique().tolist()) - set(relevant_patients))

        if len(relevant_data.index) == 0:
            logger.info(f"Not enough data for {type_of_dataset}, skipping {name}...")
            return -1, None, [], pd.DataFrame()
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
            return -1, None, [], pd.DataFrame()
        if save_data:
            dataset_file_name = self._get_filename_from_dataset_config(dataset_to_create, type_of_dataset)
            feature_path = os.path.join(self.prepared_data_dir, f"{dataset_file_name}_features.pkl")
            contained_patients_path = os.path.join(self.prepared_data_dir, f"{dataset_file_name}_patients.pkl")
            self._save_file(dataset, feature_path, True)
            self._save_file(relevant_patients, contained_patients_path, True)
            
            if type_of_dataset == "test":
                
                relevant_path = os.path.join(self.prepared_data_dir, f"{dataset_file_name}_relevant.pkl")
                patients_to_remove_path = os.path.join(self.prepared_data_dir, f"{dataset_file_name}_patients_to_remove.pkl")
                
                self._save_file(patients_to_remove, patients_to_remove_path, True)
                self._save_file(relevant_data, relevant_path, True)
        
        if type_of_dataset != "test":
            return 0, dataset, [], pd.DataFrame()
        else:
            return 0, dataset, patients_to_remove, relevant_data

    @staticmethod
    def _get_dataset_config_from_filename(filename):
        split = filename.split("_")
        name = split[0]
        dataset_features = split[1].split("+")
        dataset_config = {
            "name": name,
            "features": dataset_features
        }
        return dataset_config













