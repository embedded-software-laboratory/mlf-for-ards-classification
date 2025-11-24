import logging
import os
import pickle
import sys
from multiprocessing import Pool
from typing import Union, Any

import numpy as np

from processing.ad_algorithms.torch_utils import split_patients, check_directory
from processing.processing_utils import prepare_multiprocessing

import pandas as pd

logger = logging.getLogger(__name__)


class BaseAnomalyDetector:
    """
    Base class for anomaly detection algorithms.

    Provides:
    - common IO helpers (save/load)
    - multiprocessing orchestration
    - anomaly counting and aggregation utilities
    - generic handling strategies for detected anomalies

    Subclasses must implement algorithm specific methods:
    - _prepare_data, _train_ad_model, _predict, _use_prediction, _create_meta_data_preparation, _load_prepared_data
    """

    def __init__(self, **kwargs):
        logger.info("Initializing BaseAnomalyDetector...")
        self.trainable = False
        self.saved_anomaly_data = False
        self.anomaly_data_dir = None
        self.can_predict_value = False
        self.patient_level_anomaly_detection = False
        self.name = None
        self.type = None
        self.model = None
        self.columns_to_check = None
        self.database = None
        self.fix_algorithm = None
        self.handling_strategy = None
        self.anomaly_threshold = None
        self.max_processes = None
        self.needs_full_data = False
        self.anomaly_counts = None
        self.prepared_data_dir = None
        self.anomaly_data_dir = None
        self.supported_stages = []
        self.active_stages = []
        self.columns_not_to_check = ["patient_id", "time", "ards", "chest-injury", "sepsis", "xray", "pneumonia", "pulmonary-edema", "hypervolemia", "heart-failure", "mech-vent", "cardiac-pulmonary-edema", "gender", "height", "weight", "burn-trauma"]

        # Apply kwargs to instance if key exists
        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)
                logger.debug(f"Set attribute from kwargs: {key}={value}")

        # Validate active stages against supported stages
        for stage in list(self.active_stages):
            if stage not in self.supported_stages:
                logger.info(f"Stage '{stage}' is not supported by algorithm '{self.name}'. Removing it from active_stages.")
                self.active_stages.remove(stage)

        logger.info(f"Using up to {self.max_processes} processes for algorithm '{self.name}'. Active stages: {self.active_stages}")
        self.meta_data = None


    @staticmethod
    def _calculate_anomaly_count_full(anomaly_df: pd.DataFrame, relevant_data: pd.DataFrame,
                                      relevant_columns: list[str]) \
            -> dict[str, dict[str, int]]:
        """
        Count anomalies per column over the full dataset (not per-step).
        """
        logger.debug("Calculating full anomaly counts for relevant columns")
        anomaly_count_dict = {}

        for column in relevant_columns:
            if column in relevant_data.columns:
                total_value = relevant_data[column].count()
                anomaly_count_dict[column] = {
                    "anomaly_count": int(anomaly_df[column].sum()),
                    "total_data": int(total_value)
                }
                logger.debug(f"Column '{column}': anomalies={anomaly_count_dict[column]['anomaly_count']}, total={total_value}")

        logger.info(f"Calculated anomaly counts for {len(anomaly_count_dict)} columns")
        return anomaly_count_dict

    @staticmethod
    def _save_file(obj_to_save, file_path: str, overwrite: bool = False) -> None:
        """
            Saves the object to a .pkl file.

            Args:
                obj_to_save (object): The object to be saved.
                file_path (str): The path where the object will be saved.
                overwrite (bool): Whether to overwrite the file if it already exists.
        """
        logger.debug(f"Saving object to file: {file_path} (overwrite={overwrite})")
        if os.path.exists(file_path) and not overwrite:
            logger.info(f"File {file_path} already exists. Skipping save.")
            return
        elif os.path.exists(file_path) and overwrite:
            logger.info(f"File {file_path} already exists. Overwriting.")
        with open(file_path, "wb") as f:
            pickle.dump(obj_to_save, f)
        logger.debug(f"Saved object to {file_path}")

    @staticmethod
    def _get_first_and_last_patient_id_for_name( dataframe: pd.DataFrame) -> (str, str):
        """
        Helper to derive a compact filename suffix from the first and last patient ids
        in a dataframe.
        """
        patient_ids = dataframe["patient_id"].unique().tolist()
        first_patient_id = str(patient_ids[0]).replace(".", "_")
        last_patient_id = str(patient_ids[-1]).replace(".", "_")
        logger.debug(f"First patient id: {first_patient_id}, Last patient id: {last_patient_id}")
        return first_patient_id, last_patient_id

    def _save_anomaly_df(self, anomaly_df: pd.DataFrame) -> str:
        """
            Saves the anomaly DataFrame to a file and returns the path.
        """
        logger.info("Saving detected anomaly dataframe to disk...")
        self.saved_anomaly_data = True

        first_patient_id, last_patient_id = self._get_first_and_last_patient_id_for_name(anomaly_df)
        anomaly_df_path = f"{self.anomaly_data_dir}/anomaly_data_{self.name}_{first_patient_id}_{last_patient_id}.pkl"
        self._save_file(anomaly_df, anomaly_df_path, True)
        logger.info(f"Anomaly dataframe saved to {anomaly_df_path}")
        return anomaly_df_path

    # Algorithm-specific hooks - must be implemented in subclasses
    def _train_ad_model(self, data_training, data_validation, **kwargs):
        logger.debug("Base._train_ad_model called - must be implemented in subclass")
        raise NotImplementedError()

    def _predict(self, dataframe: pd.DataFrame, **kwargs) -> dict:
        logger.debug("Base._predict called - must be implemented in subclass")
        raise NotImplementedError()


    def _prepare_data(self, dataframe: pd.DataFrame, save_data: bool = False, overwrite: bool = False) -> dict:
        logger.debug("Base._prepare_data called - must be implemented in subclass")
        raise NotImplementedError()

    def _use_prediction(self, prediction: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Base._use_prediction called - must be implemented in subclass")
        raise NotImplementedError()

    def _create_meta_data_preparation(self, test_data: pd.DataFrame) -> dict:
        logger.debug("Base._create_meta_data_preparation called - must be implemented in subclass")
        raise NotImplementedError()

    def _load_prepared_data(self, storage_info: str, type_of_dataset: str) -> Any:
        logger.debug("Base._load_prepared_data called - must be implemented in subclass")
        raise NotImplementedError()


    def execute_handler(self, process_pool_data_list: list[pd.DataFrame],  patients_per_process: int, no_multi_processing: bool = False) -> (list[pd.DataFrame], int, pd.DataFrame):
        """
        Entry point used by the processing pipeline to run the anomaly algorithm.
        Depending on configuration it will execute the pipeline single-threaded or with multiprocessing.
        Returns the prepared process_pool_data_list, number of jobs and the fixed dataframe (if fix stage active).
        """
        logger.info("execute_handler called")
        if self.needs_full_data or no_multi_processing:
            logger.info("Running single-process execution (needs_full_data or no_multi_processing=True)")
            fixed_df = self.execute_single(self.active_stages, process_pool_data_list, patients_per_process, no_multi_processing)
        else:
            logger.info("Running multiprocessed execution")
            fixed_df = self.execute_multiprocessing(self.active_stages, process_pool_data_list, patients_per_process, no_multi_processing)
        # Prepare data for the next step after anomaly processing
        process_pool_data_list, n_jobs = prepare_multiprocessing(fixed_df, patients_per_process, self.max_processes)
        logger.info(f"execute_handler finished - prepared {n_jobs} jobs for next processing step")
        return process_pool_data_list, n_jobs, fixed_df


    def execute_single(self, stages, process_pool_data_list: list[pd.DataFrame],  patients_per_process: int, no_multiprocessing: bool = False) -> pd.DataFrame:
        """
        Executes all active stages in a single process.
        Stages: prepare, train, predict, fix
        """
        logger.info("Starting single-process execution of anomaly detector")
        dataframe = pd.concat(process_pool_data_list, ignore_index=True).reset_index(drop=True)
        logger.debug(f"Input dataframe shape: {dataframe.shape}")
        prepared_dict = {"train": None, "val": None, "test": None}
        anomaly_result_dict = {"anomaly_df": None, "anomaly_count": {}}
        fixed_df = pd.DataFrame(columns=dataframe.columns)
        logger.info(f"Executing stages {stages}")
        for stage in stages:
            logger.debug(f"Processing stage '{stage}'")
            if stage == "prepare":
                prepared_dict = self._prepare_data(dataframe, save_data=True, overwrite=False)
                meta_data_preparation = self._create_meta_data_preparation(prepared_dict["test"])
                self._save_file(meta_data_preparation, self.prepared_data_dir+"/meta_data_preparation_test.json", True)
                logger.info("Preparation stage finished and metadata saved")

            elif stage == "train" and self.trainable:
                logger.info("Training stage started")
                if prepared_dict["train"] is None or prepared_dict["val"] is None or prepared_dict["train"].empty or prepared_dict["val"].empty:
                    logger.info("Loading prepared data from storage for training/validation")
                    train_data = self._load_prepared_data(self.prepared_data_dir, "train")
                    val_data = self._load_prepared_data(self.prepared_data_dir, "val")
                    prepared_dict["train"] = train_data
                    prepared_dict["val"] = val_data
                self._train_ad_model(prepared_dict["train"], prepared_dict["val"])
                logger.info("Training stage finished")

            elif stage == "predict":
                logger.info("Prediction stage started")
                if not prepared_dict["test"]:
                    test_data = self._load_prepared_data(self.prepared_data_dir, "test")
                    prepared_dict["test"] = test_data
                    logger.debug("Loaded prepared test data from storage")
                anomaly_result_dict = self._predict(prepared_dict["test"])
                anomaly_counts = self.finalize_anomaly_counts_single(anomaly_result_dict["anomaly_count"])
                anomaly_df_meta_data = {
                    "contained_patients": anomaly_result_dict["anomaly_df"]["patient_id"].unique().tolist(),
                }
                self._save_file(anomaly_df_meta_data, self.anomaly_data_dir+"/meta_data_anomaly_df.json", True)
                self.anomaly_counts = anomaly_counts
                logger.info("Prediction stage finished")

            elif stage == "fix":
                logger.info("Fix stage started")
                # TODO fix patient_df_to_fix only containing data used for ad
                if not prepared_dict["test"]:
                    data_to_fix = dataframe
                else:
                    data_to_fix = prepared_dict["test"]
                if anomaly_result_dict["anomaly_df"] is None:
                    anomalies = self._load_stored_anomalies(self.anomaly_data_dir)
                else:
                    anomalies = anomaly_result_dict["anomaly_df"]

                fixed_df = self._handle_anomalies(anomalies, data_to_fix, no_multi_processing=no_multiprocessing)
                logger.info("Fix stage finished")

        if not "fix" in self.active_stages:
            logger.info("No fixing stage in active_stages. Exiting process as no data can be forwarded.")
            sys.exit(0)

        logger.debug(f"Single execution finished - fixed dataframe shape: {fixed_df.shape}")
        return  fixed_df

    def execute_multiprocessing(self, stages, process_pool_data_list: list[pd.DataFrame], patients_per_process: int, no_multi_processing: bool = False) -> pd.DataFrame:
        """
        Executes active stages using multiprocessing where appropriate.
        """
        logger.info("Starting multiprocess execution of anomaly detector")
        logger.info(f"Active stages: {stages}")
        anomaly_result_list = []
        prepared_data_list = []
        fixed_df = pd.DataFrame(columns=process_pool_data_list[0].columns)
        for stage in stages:
            logger.debug(f"Processing stage '{stage}' with multiprocessing")
            if stage == "prepare":
                logger.info("Prepare stage: splitting and preparing data in parallel")
                with Pool(processes=self.max_processes) as pool:
                    prepared_data_list = pool.starmap(self._prepare_data, [(dataframe, True, False) for dataframe in process_pool_data_list])
                    test_dfs = [result_dict["test"] for result_dict in prepared_data_list]
                    test_df = pd.concat(test_dfs, ignore_index=True).reset_index(drop=True)
                    meta_data_preparation = self._create_meta_data_preparation(test_df)
                    self._save_file(meta_data_preparation, self.prepared_data_dir+"/meta_data_preparation_test.json", True)
                    logger.info("Parallel preparation complete and metadata saved")


            elif stage == "train" and self.trainable:
                logger.info("Train stage: running training in parallel")
                if not prepared_data_list:
                    logger.info("Loading prepared training/validation data from storage")
                    train_data = self._load_prepared_data(self.prepared_data_dir, "train")
                    train_data_list = prepare_multiprocessing(train_data, patients_per_process, self.max_processes)

                    val_data = self._load_prepared_data(self.prepared_data_dir, "val")
                    val_data_list = prepare_multiprocessing(val_data, patients_per_process, self.max_processes)
                    prepared_data_list = [{"train": train_data, "val": val_data} for train_data, val_data in zip(train_data_list, val_data_list)]
                with Pool(processes=self.max_processes) as pool:
                    train_data_list = [(result_dict["train"], result_dict["val"]) for result_dict in prepared_data_list]
                    pool.starmap(self._train_ad_model, [(train, val) for train, val in train_data_list])
                    logger.info("Parallel training complete")

            elif stage == "predict":
                logger.info("Predict stage: running prediction in parallel")
                if not prepared_data_list:
                    test_data = self._load_prepared_data(self.prepared_data_dir, "test")
                    test_data_list, _ = prepare_multiprocessing(test_data, patients_per_process, self.max_processes)
                    prepared_data_list = [{"test": test_data} for test_data in test_data_list]
                with Pool(processes=self.max_processes) as pool:
                    predict_data_list = [result_dict["test"] for result_dict in prepared_data_list]
                    anomaly_result_list = pool.starmap(self._predict, [(dataframe, )for dataframe in predict_data_list])
                anomaly_count_list = [anomaly_result["anomaly_count"] for anomaly_result in anomaly_result_list]
                contained_patients = [anomaly_result["anomaly_df"]["patient_id"].unique().tolist() for anomaly_result in anomaly_result_list]
                contained_patients = [item for sublist in contained_patients for item in sublist]
                meta_data_anomaly_df = {
                    "contained_patients": contained_patients,
                }
                self._save_file(meta_data_anomaly_df, self.anomaly_data_dir+"/meta_data_anomaly_df.json", True)

                anomaly_count = self.finalize_anomaly_counts_multiprocessing(anomaly_count_list)
                self.anomaly_counts = anomaly_count
                logger.info("Parallel prediction complete and anomaly counts aggregated")


            elif stage == "fix":
                logger.info("Fix stage: handling anomalies and producing fixed dataset")
                # TODO fix patient_df_to_fix only containing data used for ad
                if not anomaly_result_list:
                    anomaly_result = self._load_stored_anomalies(self.anomaly_data_dir)
                else:
                    anomaly_result = pd.concat([anomaly_result["anomaly_df"] for anomaly_result in anomaly_result_list], ignore_index=True).reset_index(drop=True)
                if prepared_data_list:
                    patient_df_to_fix = pd.concat([result_dict["test"] for result_dict in prepared_data_list], ignore_index=True).reset_index(drop=True)
                else:
                    patient_df_to_fix = pd.concat(process_pool_data_list, ignore_index=True).reset_index(drop=True)

                logger.info("Starting handling of detected anomalies")
                fixed_df = self._handle_anomalies(anomaly_result, patient_df_to_fix, no_multi_processing=no_multi_processing)
                logger.info("Finished handling anomalies in parallel")


        if not "fix" in self.active_stages:
            logger.info("No fixing stage in active_stages. Exiting process as no data can be forwarded.")
            sys.exit(0)
        logger.debug(f"Multiprocess execution finished - fixed dataframe shape: {fixed_df.shape}")
        return fixed_df


    @staticmethod
    def finalize_anomaly_counts_multiprocessing(anomaly_count_list: list) -> dict[str, dict[str, Union[int, float]]]:
        """
        Aggregates anomaly counts produced by multiple processes.
        """
        logger.debug("Finalizing anomaly counts from multiprocessing results")
        anomaly_counts = {}
        for anomaly_count in anomaly_count_list:
            if not anomaly_counts:
                for key, value in anomaly_count.items():
                    total_anomalies_name = key + "_total_anomalies"
                    total_data_name = key + "_total_data"
                    percentage_anomalies_name = key + "_percentage_anomalies"
                    anomaly_counts = {
                        key: {
                            total_anomalies_name: int(value["anomaly_count"]),
                            total_data_name: int(value["total_data"]),
                            percentage_anomalies_name: float(value["anomaly_count"]) / float(value["total_data"]) if value["total_data"] else 0.0
                        }
                    }
            else:
                for key, value in anomaly_count.items():
                    total_anomalies_name = key + "_total_anomalies"
                    total_data_name = key + "_total_data"
                    percentage_anomalies_name = key + "_percentage_anomalies"

                    if key in anomaly_counts.keys():
                        data = anomaly_counts[key]
                        anomaly_counts[key] = {
                            total_anomalies_name: data[total_anomalies_name] + int(value["anomaly_count"]),
                            total_data_name: data[total_data_name] + int(value["total_data"]),
                            percentage_anomalies_name: (data[total_anomalies_name] + int(value["anomaly_count"])) / (data[total_data_name] + int(value["total_data"])) if (data[total_data_name] + int(value["total_data"])) else 0.0
                        }
                    else:
                        anomaly_counts[key] = {
                            total_anomalies_name: int(value["anomaly_count"]),
                            total_data_name: int(value["total_data"]),
                            percentage_anomalies_name: float(value["anomaly_count"]) / float(value["total_data"]) if value["total_data"] else 0.0
                        }
        logger.info("Aggregated multiprocessing anomaly counts")
        return anomaly_counts


    @staticmethod
    def finalize_anomaly_counts_single(anomaly_counts: dict[str, dict[str, Union[int, float]]]) -> dict[str, dict[str, Union[int, float]]]:
        """
        Finalizes the anomaly count dictionary produced by a single process.
        """
        logger.debug("Finalizing anomaly counts for single-process run")
        finished_anomaly_count_dict = {}
        for key, value in anomaly_counts.items():
            total_anomalies_name = key + "_total_anomalies"
            total_data_name = key + "_total_data"
            percentage_anomalies_name = key + "_percentage_anomalies"
            finished_anomaly_count_dict[key] = {
                total_anomalies_name: int(value["anomaly_count"]),
                total_data_name: int(value["total_data"]),
                percentage_anomalies_name: float(value["anomaly_count"]) / float(value["total_data"]) if value["total_data"] else 0.0
            }
            logger.debug(f"Finalized counts for '{key}': {finished_anomaly_count_dict[key]}")
        logger.info("Finalized single-process anomaly counts")
        return finished_anomaly_count_dict

    @staticmethod
    def _load_anomaly_df(anomaly_df_path: str, filename: str) -> pd.DataFrame:
        """
        Helper to load a single anomaly dataframe from disk.
        """
        full_path = os.path.join(anomaly_df_path, filename)
        logger.debug(f"Loading anomaly dataframe from {full_path}")
        df = pd.read_pickle(full_path)
        temp = df.drop(columns=["patient_id", "time"])
        if "ards" in list(temp.columns):
            logger.warning(f"'ards' column present in anomaly file {full_path} - this is unexpected")
        return df

    def _load_stored_anomalies(self, detected_anomalies_path: str) -> pd.DataFrame:
        """
        Loads all stored anomaly files from detected_anomalies_path using multiprocessing to speed up IO.
        Returns the concatenated dataframe and writes combined metadata.
        """
        logger.info(f"Loading stored anomalies from {detected_anomalies_path}")
        detected_anomalies_df_list = []
        existing_anomaly_files = [f for f in os.listdir(detected_anomalies_path) if f.endswith(".pkl")]
        existing_meta_data_files = [f for f in os.listdir(detected_anomalies_path) if f.endswith(".json")]
        with Pool(processes=self.max_processes) as pool:
            detected_anomalies_df_list = pool.starmap(self._load_anomaly_df, [(detected_anomalies_path, file) for file in existing_anomaly_files])

        detected_anomalies_df = pd.concat(detected_anomalies_df_list, ignore_index=True).reset_index(drop=True)

        meta_data_list = []
        for file in existing_meta_data_files:
            full_path = os.path.join(detected_anomalies_path, file)
            with open(full_path, "rb") as f:
                meta_data_list.append(pickle.load(f))
                logger.debug(f"Loaded anomaly metadata from {full_path}")
        meta_data = {"contained_patients": []}
        for meta_data_dataset in meta_data_list:
            if meta_data_dataset.get("contained_patients"):
                new_contained_patients = list(set(meta_data_dataset["contained_patients"] + meta_data["contained_patients"]))
                meta_data["contained_patients"] = new_contained_patients
        if detected_anomalies_df.empty or detected_anomalies_df is None:
            logger.info(f"No anomaly files found in {detected_anomalies_path}. Exiting.")
            sys.exit(0)
        logger.info(f"Loaded {len(detected_anomalies_df)} anomaly rows across {len(existing_anomaly_files)} files")
        return detected_anomalies_df



    def _handle_anomalies_patient(self, anomaly_df: pd.DataFrame, relevant_data: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        """
            This function is used to clean the anomalies in the data. It is applied for each patient separately.
        """
        logger.debug("Handling anomalies for a single patient")
        anomaly_df = anomaly_df.copy()
        anomaly_df.drop(columns=["patient_id", "time"], inplace=True)
        if anomaly_df.empty or original_data.empty:
            logger.info("No data to fix for this patient. Returning empty dataframe with original columns.")
            return pd.DataFrame(columns=original_data.columns)


        if self.handling_strategy == "delete_value":
            logger.debug("Handling strategy: delete_value")
            fixed_df = self._delete_value(anomaly_df, relevant_data)
        elif self.handling_strategy == "delete_than_impute":
            logger.debug("Handling strategy: delete_than_impute")
            fixed_df = self._delete_than_impute(anomaly_df, relevant_data)
        elif self.handling_strategy == "delete_row_if_any_anomaly":
            logger.debug("Handling strategy: delete_row_if_any_anomaly")
            fixed_df = self._delete_row_if_any_anomaly(anomaly_df, relevant_data)
        elif self.handling_strategy == "delete_row_if_many_anomalies":
            logger.debug("Handling strategy: delete_row_if_many_anomalies")
            fixed_df = self._delete_row_if_many_anomalies(anomaly_df, relevant_data)
        elif self.handling_strategy == "use_prediction" and self.can_predict_value:
            logger.debug("Handling strategy: use_prediction (prediction supported)")
            fixed_df = self._use_prediction(anomaly_df, relevant_data)
        elif self.handling_strategy == "use_prediction" and not self.can_predict_value:
            logger.error(f"Algorithm {self.type} does not support prediction as a handling strategy.")
            raise NotImplementedError(f"Algorithm {self.type} does not support prediction as a handling strategy for anomalies.")
        else:
            logger.error(f"Unknown fixing strategy '{self.handling_strategy}'")
            raise ValueError(f"Unknown fixing strategy {self.handling_strategy}")
        finished_df = original_data.copy()
        finished_df.update(fixed_df)
        logger.debug("Patient anomaly handling complete")
        return finished_df

    @staticmethod
    def _fix_anomaly_df(anomaly_df: pd.DataFrame, starting_data: pd.DataFrame) -> pd.DataFrame:
        """
            Ensure anomaly dataframe has an entry for every timestamp in starting_data.
            Missing entries are assumed to be False.
        """
        logger.debug("Normalizing anomaly dataframe to cover all timestamps of the starting data")
        anomaly_df_fixed = anomaly_df.copy(deep=True)
        all_timestamps = starting_data[["patient_id", "time"]].drop_duplicates(subset=["patient_id", "time"], keep="first")
        anomaly_df_fixed = pd.merge(all_timestamps, anomaly_df_fixed, how="outer", on=["patient_id", "time"])
        anomaly_df_fixed.fillna(False, inplace=True)
        anomaly_df_fixed = anomaly_df_fixed.sort_values(by=["time"], ascending=True).reset_index(drop=True)

        logger.debug(f"Anomaly df normalized: resulting shape {anomaly_df_fixed.shape}")
        return anomaly_df_fixed

    def _calculate_anomaly_count(self, anomaly_df: pd.DataFrame, relevant_data: pd.DataFrame,
                                 anomaly_count_dict: dict = None) -> dict[str, dict[str, int]]:
        """
            Count anomalies either per-step or full depending on patient_level_anomaly_detection flag.
        """
        logger.debug("Calculating anomaly count (dispatcher)")
        relevant_columns = [column for column in anomaly_df.columns if column not in self.columns_not_to_check]
        if self.patient_level_anomaly_detection:
            logger.debug("Using per-step anomaly counting")
            anomaly_count_dict = self._calculate_anomaly_count_per_step(anomaly_df, relevant_data, relevant_columns, anomaly_count_dict)
        else:
            logger.debug("Using full anomaly counting")
            anomaly_count_dict = self._calculate_anomaly_count_full(anomaly_df, relevant_data, relevant_columns)
        return anomaly_count_dict

    @staticmethod
    def _calculate_anomaly_count_per_step(anomaly_df: pd.DataFrame, relevant_data: pd.DataFrame, relevant_columns: list[str],
                                                anomaly_count_dict: dict = None) -> dict[str, dict[str, int]]:
        logger.debug("Calculating anomaly counts per step (patient-level)")
        if anomaly_count_dict is None:
            anomaly_count_dict = {}
        for column in relevant_columns:
            value_count = int(relevant_data[column].count())
            if column == "patient_id" or column == "time":
                continue
            anomaly_count = int(anomaly_df[column].sum())
            if column in anomaly_count_dict.keys():
                new_anomaly_count = anomaly_count_dict[column]["anomaly_count"] + anomaly_count
                new_value_count = anomaly_count_dict[column]["total_data"] + value_count
                anomaly_count_dict[column] = {"total_data": new_value_count,
                                              "anomaly_count": new_anomaly_count}
            else:
                anomaly_count_dict[column] = {"total_data": value_count,
                                              "anomaly_count": anomaly_count}
            logger.debug(f"Column '{column}': anomaly_count={anomaly_count_dict[column]['anomaly_count']}, total_data={anomaly_count_dict[column]['total_data']}")
        return anomaly_count_dict




    def _handle_anomalies(self, detected_anomalies_df : pd.DataFrame, original_data: pd.DataFrame, save_data: bool =True, save_path: str = None, no_multi_processing: bool = False) -> pd.DataFrame:
        """
        Main entry to apply anomaly fixes across patients (multiprocessing supported).
        """
        logger.info("Handling anomalies for dataset")
        if original_data.empty or original_data is None:
            logger.info("No data to fix. Exiting...")
            sys.exit(0)

        if not save_path:
            save_path = os.path.join(self.anomaly_data_dir, "fixed_data")
        check_directory(save_path)

        anomaly_df_patients = detected_anomalies_df["patient_id"].unique().tolist()
        original_data_patients = original_data["patient_id"].unique().tolist()
        contained_patients = list(set(anomaly_df_patients).intersection(original_data_patients))
        detected_anomalies_df = detected_anomalies_df[detected_anomalies_df["patient_id"].isin(contained_patients)].reset_index(drop=True)
        original_data_df = original_data[original_data["patient_id"].isin(contained_patients)].reset_index(drop=True)
        relevant_data = original_data[detected_anomalies_df.columns]

        if detected_anomalies_df.empty or detected_anomalies_df is None:
            logger.info(f"No overlapping patients found between stored anomalies and data to fix..")
            sys.exit(0)
        if original_data_df.empty or original_data_df is None:
            logger.info(f"No overlapping patients found between stored anomalies and data to fix.")
            sys.exit(0)

        detected_anomalies_df = self._fix_anomaly_df(detected_anomalies_df, relevant_data)

        if not self.columns_to_check:
            # TODO handle setting columns to check
            logger.debug("No explicit columns_to_check provided - defaulting to all relevant columns")
            pass
        if not self.anomaly_counts:
            relevant_columns = [column for column in detected_anomalies_df.columns if column not in self.columns_not_to_check]
            anomaly_counts_dict = self._calculate_anomaly_count_full(detected_anomalies_df, relevant_data, relevant_columns)
            self.anomaly_counts = self.finalize_anomaly_counts_single(anomaly_counts_dict)
            logger.info("Computed anomaly statistics for dataset")

        original_data_df_list = [y for x,y in original_data_df.groupby("patient_id")]
        anomaly_df_list = [y for x,y in detected_anomalies_df.groupby("patient_id")]
        relevant_data_df_list = [y for x,y in relevant_data.groupby("patient_id")]
        logger.debug(f"Prepared {len(original_data_df_list)} patient groups for handling")

        if not no_multi_processing:
            logger.info(f"Handling anomalies with multiprocessing using {self.max_processes} processes")
            with Pool(processes=self.max_processes) as pool:
                fixed_dfs = pool.starmap(self._handle_anomalies_patient,
                                         [(anomaly_df_list[i], relevant_data_df_list[i], original_data_df_list[i])
                                                                          for i in range(len(original_data_df_list))])
        else:
            logger.info("Handling anomalies without multiprocessing")
            fixed_dfs = []
            for i in range(len(original_data_df_list)):
                fixed_df = self._handle_anomalies_patient(anomaly_df_list[i], relevant_data_df_list[i], original_data_df_list[i])
                fixed_dfs.append(fixed_df)
        fixed_df = pd.concat(fixed_dfs, ignore_index=True).reset_index(drop=True)

        if save_data:
            fixed_data_path = os.path.join(save_path, f"fixed_data_{self.name}_{self.handling_strategy}_{self.fix_algorithm}.pkl")
            logger.info(f"Saving fixed data to {fixed_data_path}")
            self._save_file(fixed_df, fixed_data_path, True)
        logger.info("Anomaly handling complete for dataset")
        return fixed_df

    @staticmethod
    def _delete_value(anomaly_df: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
            Replace anomalous values with NaN (value-level deletion).
        """
        logger.debug("Deleting anomalous values (masking to NaN)")
        dataframe = dataframe.mask(anomaly_df)
        return dataframe

    def _delete_than_impute(self, anomaly_df: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
            Delete anomalous values and then impute/fix the deleted values according to fix_algorithm.
        """
        logger.debug("Deleting anomalous values and imputing/fixing afterwards")
        nan_mask = dataframe.isna()
        dataframe = dataframe.mask(anomaly_df)
        dataframe = self._fix_deleted(dataframe, nan_mask=nan_mask)
        return dataframe

    @staticmethod
    def _delete_row_if_any_anomaly(anomaly_df: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
            Delete entire row if any anomaly present in that row.
        """
        logger.debug("Deleting rows that contain any anomalies")
        dataframe = dataframe.mask(anomaly_df)
        index_to_drop = []
        for index, row in dataframe.iterrows():
            if row.isnull().any():
                index_to_drop.append(index)
        dataframe = dataframe.drop(index_to_drop).reset_index(drop=True)
        logger.debug(f"Dropped {len(index_to_drop)} rows containing anomalies")
        return dataframe

    def _delete_row_if_many_anomalies(self, anomaly_df: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
            Delete entire row if the proportion of anomalous values exceeds anomaly_threshold.
        """
        logger.debug("Deleting rows with many anomalies (threshold based)")
        n_columns = len(dataframe.columns)
        n_anomalies = anomaly_df.sum(axis=1)
        index_to_drop = []
        for i in range(len(dataframe)):
            if n_anomalies[i]/n_columns > self.anomaly_threshold:
                index_to_drop.append(i)

        anomaly_df = anomaly_df.drop(index_to_drop).reset_index(drop=True)
        dataframe = dataframe.drop(index_to_drop).reset_index(drop=True)
        nan_mask = dataframe.isna()
        dataframe = dataframe.mask(anomaly_df)
        dataframe = self._fix_deleted(dataframe, nan_mask=nan_mask)
        logger.debug(f"Dropped {len(index_to_drop)} rows for exceeding anomaly threshold")
        return dataframe

    @staticmethod
    def _add_anomaly_score(anomaly_df: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Add binary anomaly indicator columns next to original columns.
        """
        logger.debug("Adding anomaly score columns to dataframe")
        for column in dataframe.columns:
            if column in anomaly_df.columns:
                dataframe[column + "_anomaly"] = anomaly_df[column]
        return dataframe

    def _fix_deleted(self, dataframe: pd.DataFrame, nan_mask) -> pd.DataFrame:
        """
        Apply chosen fix algorithm to fill previously deleted values.
        """
        logger.debug(f"Applying fix_algorithm='{self.fix_algorithm}' to fill deleted values")
        if self.fix_algorithm == "forward":
            dataframe = dataframe.fillna(method="ffill")
        elif self.fix_algorithm == "backward":
            dataframe = dataframe.fillna(method="bfill")
        elif self.fix_algorithm == "interpolate":
            dataframe = dataframe.interpolate(method="linear", limit_direction="both")
        else:
            logger.error("Invalid fix_algorithm provided for Anomaly detection")
            raise ValueError("Invalid fix_algorithm for Anomaly detection")
        dataframe = dataframe.mask(nan_mask)
        logger.debug("Fixing of deleted values complete")
        return dataframe


    def create_meta_data(self) -> dict:
        """
        Build and return a metadata dictionary describing the anomaly detection run/settings.
        """
        logger.info("Creating anomaly detector metadata")
        meta_data_dict = {
            "name": self.name,
            "anomaly_detection_algorithm": None,
            "anomaly_handling_strategy": self.handling_strategy,
            "anomaly_fixing_algorithm": self.fix_algorithm,
            #"columns_checked": self.columns_to_check,
            "anomaly_statistics":  self.anomaly_counts,
            "algorithm_specific_settings": None,
        }

        if self.handling_strategy == "delete_row_if_many_anomalies":
            meta_data_dict["anomaly_threshold"] = self.anomaly_threshold
        else:
            meta_data_dict["anomaly_threshold"] = None
        if self.saved_anomaly_data:
            meta_data_dict["anomaly_data_dir"] = self.anomaly_data_dir

        logger.debug(f"Meta data created: {meta_data_dict}")
        return meta_data_dict




