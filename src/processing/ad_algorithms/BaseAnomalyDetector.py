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


    def __init__(self, **kwargs):
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
        self.columns_not_to_check = ["patient_id", "time", "ards", "chest-injury", "sepsis", "xray", "pneumonia", "pulmonary-edema", "hypervolemia", "heart-failure", "mech-vent"]
        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)
        for stage in self.active_stages:
            if stage not in self.supported_stages:
                logger.info(f"Stage {stage} is not supported by the algorithm {self.name}. Supported stages are: {self.supported_stages}. Removing stage {stage} from the active stages.")
                self.active_stages.remove(stage)
        logger.info(f"Using up to {self.max_processes} processes for the algorithm {self.name}.")
        self.meta_data = None



    def run(self,  dataframe_detection: pd.DataFrame, job_count: int, total_jobs: int) -> (pd.DataFrame, dict[str, dict[str, int]]):
        """
            Runs the anomaly detection process from start to finish.

            Args:
                dataframe_detection (pd.DataFrame): The input DataFrame containing the data to be processed.
                job_count (int): The current job count (Irrelevant for approaches that need the full dataset).
                total_jobs (int): The total number of jobs (Irrelevant for approaches that need the full dataset).

            Returns:
                pd.DataFrame: The processed DataFrame with anomalies handled.
                dict: A dictionary containing the anomaly counts for each column as well as the total amount of data present for each column.

        """
        raise NotImplementedError()

    @staticmethod
    def _calculate_anomaly_count_full(anomaly_df: pd.DataFrame, relevant_data: pd.DataFrame,
                                      relevant_columns: list[str]) \
            -> dict[str, dict[str, int]]:
        anomaly_count_dict = {}

        for column in relevant_columns:
            if column in relevant_data.columns:
                total_value = relevant_data[column].count()
                anomaly_count_dict[column] = {
                    "anomaly_count": anomaly_df[column].sum(),
                    "total_data": total_value
                }

        return anomaly_count_dict
    @staticmethod
    def _save_file(obj_to_save, file_path: str, overwrite: bool = False) -> None:
        """
            Saves the object to a .pkl file.

            Args:
                obj_to_save (object): The object to be saved.
                file_path (str): The path where the object will be saved.
                overwrite (bool): Whether to overwrite the file if it already exists.

            Returns:
                None
        """
        if os.path.exists(file_path) and not overwrite:
            logger.info(f"File {file_path} already exists. Skipping.")
            return
        elif os.path.exists(file_path) and overwrite:
            logger.info(f"File {file_path} already exists. Overwriting.")
        with open(file_path, "wb") as f:
            pickle.dump(obj_to_save, f)

    @staticmethod
    def _get_first_and_last_patient_id_for_name( dataframe: pd.DataFrame) -> (str, str):
        patient_ids = dataframe["patient_id"].unique().tolist()
        first_patient_id = str(patient_ids[0]).replace(".", "_")
        last_patient_id = str(patient_ids[-1]).replace(".", "_")
        return first_patient_id, last_patient_id

    def _save_anomaly_df(self, anomaly_df: pd.DataFrame) -> str:

        """
            Saves the anomaly DataFrame to a file.

            Args:
                anomaly_df (pd.DataFrame): The DataFrame containing the detected anomalies.

            Returns:
                str: The file path where the anomaly DataFrame is saved.
        """
        self.saved_anomaly_data = True

        first_patient_id, last_patient_id = self._get_first_and_last_patient_id_for_name(anomaly_df)
        anomaly_df_path = f"{self.anomaly_data_dir}/anomaly_data_{self.name}_{first_patient_id}_{last_patient_id}.pkl"
        self._save_file(anomaly_df, anomaly_df_path)

        return anomaly_df_path

    def _train_ad_model(self, data_training, data_validation, **kwargs):

        raise NotImplementedError()

    def _predict(self, dataframe: pd.DataFrame, **kwargs) -> dict:
        raise NotImplementedError()

    def _predict_proba(self):
        raise NotImplementedError()

    def _prepare_data(self, dataframe: pd.DataFrame, save_data: bool = False, overwrite: bool = False) -> dict:
        raise NotImplementedError()

    def _use_prediction(self, prediction: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def _create_meta_data_preparation(self, test_data: pd.DataFrame) -> dict:
        raise NotImplementedError()

    def _load_prepared_data(self, path: str, type_of_dataset: str) -> Any:
        raise NotImplementedError()


    def execute_handler(self, process_pool_data_list: list[pd.DataFrame],  patients_per_process: int, no_multi_processing: bool = False) -> (list[pd.DataFrame], int, pd.DataFrame):
        if self.needs_full_data or no_multi_processing:
            logger.info("Starting single execution")
            fixed_df = self.execute_single(self.active_stages, process_pool_data_list, patients_per_process, no_multi_processing)
        else:
            logger.info("Starting multiprocessed execution")
            fixed_df = self.execute_multiprocessing(self.active_stages, process_pool_data_list, patients_per_process, no_multi_processing)
        process_pool_data_list, n_jobs = prepare_multiprocessing(fixed_df, patients_per_process)
        return process_pool_data_list, n_jobs, fixed_df


    def execute_single(self, stages, process_pool_data_list: list[pd.DataFrame],  patients_per_process: int, no_multiprocessing: bool = False) -> pd.DataFrame:
        dataframe = pd.concat(process_pool_data_list, ignore_index=True).reset_index(drop=True)
        prepared_dict = {"train": None, "val": None, "test": None}
        anomaly_result_dict = {"anomaly_df": None, "anomaly_count": {}}
        fixed_df = pd.DataFrame(columns=dataframe.columns)
        logger.info(f"Executing stages {stages}")
        for stage in stages:
            if stage == "prepare":
                prepared_dict = self._prepare_data(dataframe, save_data=True, overwrite=False)
                meta_data_preparation = self._create_meta_data_preparation(prepared_dict["test"])
                self._save_file(meta_data_preparation, self.prepared_data_dir+"/meta_data_preparation_test.json", True)

            elif stage == "train" and self.trainable:
                logger.info(f"Training stage")
                if not prepared_dict["train"] or not prepared_dict["val"]:
                    logger.info("Loading data for training")
                    train_data = self._load_prepared_data(self.prepared_data_dir, "train")
                    val_data = self._load_prepared_data(self.prepared_data_dir, "val")
                    prepared_dict["train"] = train_data
                    prepared_dict["val"] = val_data
                self._train_ad_model(prepared_dict["train"], prepared_dict["val"])

            elif stage == "predict":
                if not prepared_dict["test"]:
                    test_data = self._load_prepared_data(self.prepared_data_dir, "test")
                    prepared_dict["test"] = test_data
                anomaly_result_dict = self._predict(prepared_dict["test"])
                anomaly_counts = self.finalize_anomaly_counts_single(anomaly_result_dict)
                anomaly_df_meta_data = {
                    "contained_patients": anomaly_result_dict["anomaly_df"]["patient_id"].unique().tolist(),
                }
                self._save_file(anomaly_df_meta_data, self.anomaly_data_dir+"/meta_data_anomaly_df.json", True)
                self.anomaly_counts = anomaly_counts

            elif stage == "fix":
                if not prepared_dict["test"]:
                    data_to_fix = dataframe
                else:
                    data_to_fix = prepared_dict["test"]
                if not anomaly_result_dict["anomaly_df"]:
                    anomalies = self._load_stored_anomalies(self.anomaly_data_dir)
                else:
                    anomalies = anomaly_result_dict["anomaly_df"]

                fixed_df = self._handle_anomalies(anomalies, data_to_fix, detected_anomalies_df=no_multiprocessing)

        if not "fix" in self.active_stages:
            logger.info("No fixing stage in the active stages. No data can be passed to the next module. Exiting...")
            sys.exit(0)


        return  fixed_df

    def execute_multiprocessing(self, stages, process_pool_data_list: list[pd.DataFrame], patients_per_process: int, no_multi_processing: bool = False) -> pd.DataFrame:
        logger.info(f"Active stages: {stages}")
        anomaly_result_list = []
        prepared_data_list = []
        fixed_df = pd.DataFrame(columns=process_pool_data_list[0].columns)
        for stage in stages:
            if stage == "prepare":
                with Pool(processes=self.max_processes) as pool:
                    prepared_data_list = pool.starmap(self._prepare_data, [(dataframe, True, False) for dataframe in process_pool_data_list])
                    test_dfs = [result_dict["test"] for result_dict in prepared_data_list]
                    test_df = pd.concat(test_dfs, ignore_index=True).reset_index(drop=True)
                    meta_data_preparation = self._create_meta_data_preparation(test_df)
                    self._save_file(meta_data_preparation, self.prepared_data_dir+"/meta_data_preparation_test.json", True)



            elif stage == "train" and self.trainable:
                if not prepared_data_list:
                    train_data = self._load_prepared_data(self.prepared_data_dir, "train")
                    train_data_list = prepare_multiprocessing(train_data, patients_per_process)

                    val_data = self._load_prepared_data(self.prepared_data_dir, "val")
                    val_data_list = prepare_multiprocessing(val_data, patients_per_process)
                    prepared_data_list = [{"train": train_data, "val": val_data} for train_data, val_data in zip(train_data_list, val_data_list)]
                with Pool(processes=self.max_processes) as pool:

                    train_data_list = [(result_dict["train"], result_dict["val"]) for result_dict in prepared_data_list]
                    pool.starmap(self._train_ad_model, [(train, val) for train, val in train_data_list])

            elif stage == "predict":
                if not prepared_data_list:
                    test_data = self._load_prepared_data(self.prepared_data_dir, "test")
                    test_data_list = prepare_multiprocessing(test_data, patients_per_process)
                    prepared_data_list = [{"test": test_data} for test_data in test_data_list]
                with Pool(processes=self.max_processes) as pool:
                    predict_data_list = [result_dict["test"] for result_dict in prepared_data_list]
                    anomaly_result_list = pool.starmap(self._predict, [(dataframe) for dataframe in predict_data_list])
                anomaly_count_list = [anomaly_result["anomaly_count"] for anomaly_result in anomaly_result_list]
                contained_patients = [anomaly_result["anomaly_df"]["patient_id"].unique().to_list() for anomaly_result in anomaly_result_list]
                contained_patients = [item for sublist in contained_patients for item in sublist]
                meta_data_anomaly_df = {
                    "contained_patients": contained_patients,
                }
                self._save_file(meta_data_anomaly_df, self.anomaly_data_dir+"/meta_data_anomaly_df.json", True)


                anomaly_count = self.finalize_anomaly_counts_multiprocessing(anomaly_count_list)
                self.anomaly_counts = anomaly_count


            elif stage == "fix":

                if not anomaly_result_list:
                    anomaly_result = self._load_stored_anomalies(self.anomaly_data_dir)


                else:
                    anomaly_result = pd.concat([anomaly_result["anomaly_df"] for anomaly_result in anomaly_result_list], ignore_index=True).reset_index(drop=True)
                if prepared_data_list:
                    patient_df_to_fix = pd.concat([result_dict["test"] for result_dict in prepared_data_list], ignore_index=True).reset_index(drop=True)
                else:
                    patient_df_to_fix = pd.concat(process_pool_data_list, ignore_index=True).reset_index(drop=True)

                logger.info("Starting handling")

                fixed_df = self._handle_anomalies(anomaly_result, patient_df_to_fix, no_multi_processing=no_multi_processing)
                logger.info("Finished handling")


        if not "fix" in self.active_stages:
            logger.info("No fixing stage in the active stages. No data can be passed to the next module. Exiting...")
            sys.exit(0)
        return fixed_df


    def run_full(self, process_pool_data_list: list[pd.DataFrame], n_jobs: int, patients_per_process: int)\
        -> (list[pd.DataFrame], dict[str, dict[str, Union[int, float]]], pd.DataFrame):

        if not self.needs_full_data:
            process_pool_data_list, anomaly_counts, dataframe = self.run_multiprocessing(process_pool_data_list, n_jobs)

        else:
            process_pool_data_list, anomaly_counts, dataframe = self.run_single(process_pool_data_list,
                                                                                patients_per_process)

        self.anomaly_counts = anomaly_counts

        return process_pool_data_list, n_jobs, dataframe


    def run_multiprocessing(self,  process_pool_data_list: list[pd.DataFrame], n_jobs: int) -> (list[pd.DataFrame], dict[str, dict[str, Union[int, float]]], pd.DataFrame):
        with Pool(processes=self.max_processes) as pool:
            process_pool_data_list, anomaly_count_list = pool.starmap(self.run,
                                                                      [(process_pool_data_list[i], i, n_jobs) for i in
                                                                       range(n_jobs)])
        anomaly_counts = self.finalize_anomaly_counts_multiprocessing(anomaly_count_list)
        fixed_df = pd.concat(process_pool_data_list, ignore_index=True).reset_index(drop=True)
        return process_pool_data_list,  anomaly_counts, fixed_df

    @staticmethod
    def finalize_anomaly_counts_multiprocessing(anomaly_count_list: list) -> dict[str, dict[str, Union[int, float]]]:
        anomaly_counts = {}
        for anomaly_count in anomaly_count_list:
            if not anomaly_counts:
                for key, value in anomaly_count.items():
                    total_anomalies_name = key + "_total_anomalies"
                    total_data_name = key + "_total_data"
                    percentage_anomalies_name = key + "_percentage_anomalies"
                    anomaly_counts = {
                        key: {
                            total_anomalies_name: value["anomaly_count"],
                            total_data_name: value["total_data"],
                            percentage_anomalies_name: value["anomaly_count"] / value["total_data"]
                        }
                    }
            else:
                for key, value in anomaly_count.items:

                    data = anomaly_counts[key]
                    total_anomalies_name = key + "_total_anomalies"

                    total_data_name = key + "_total_data"
                    percentage_anomalies_name = key + "_percentage_anomalies"
                    if key in anomaly_counts.keys():
                        anomaly_counts[key] = {
                            total_anomalies_name: data[total_anomalies_name] + value["anomaly_count"],
                            total_data_name: data[total_data_name] + value["total_data"],
                            percentage_anomalies_name: (data[total_anomalies_name] + value["anomaly_count"]) /
                                                       (data[total_data_name] + value["total_data"])
                        }
                    else:
                        anomaly_counts[key] = {
                            total_anomalies_name: value["anomaly_count"],
                            total_data_name: value["total_data"],
                            percentage_anomalies_name: value["anomaly_count"] / value["total_data"]
                        }
        return anomaly_counts

    def run_single(self, process_pool_data_list: list[pd.DataFrame], patients_per_process: int) -> (list[pd.DataFrame], dict[str, dict[str, Union[int, float]]], pd.DataFrame):
        dataframe = pd.concat(process_pool_data_list, ignore_index=True).reset_index(drop=True)
        fixed_df, anomaly_counts = self.run(dataframe, 0, 1)
        anomaly_counts = self.finalize_anomaly_counts_single(anomaly_counts)
        process_pool_data_list, n_jobs = prepare_multiprocessing(fixed_df, patients_per_process)
        return process_pool_data_list, anomaly_counts, fixed_df

    @staticmethod
    def finalize_anomaly_counts_single(anomaly_counts: dict[str, dict[str, Union[int, float]]]) -> dict[str, dict[str, Union[int, float]]]:
        finished_anomaly_count_dict = {}
        for key, value in anomaly_counts.items():
            total_anomalies_name = key + "_total_anomalies"
            total_data_name = key + "_total_data"
            percentage_anomalies_name = key + "_percentage_anomalies"
            finished_anomaly_count_dict[key] = {
                total_anomalies_name: value["anomaly_count"],
                total_data_name: value["total_data"],
                percentage_anomalies_name: value["anomaly_count"] / value["total_data"]
            }
        return finished_anomaly_count_dict


    def _load_anomaly_df(self, anomaly_df_path: str, filename: str) -> pd.DataFrame:

        full_path = os.path.join(anomaly_df_path, filename)

        df = pd.read_pickle(full_path)
        temp = df.drop(columns=["patient_id", "time"])

        if "ards" in list(temp.columns):
            logger.info(f"DANGER {full_path}")
        return df

    def _load_stored_anomalies(self, detected_anomalies_path: str) -> pd.DataFrame:
        logger.info(f"Path to load from {detected_anomalies_path}")
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
        meta_data = {"contained_patients": []}
        for meta_data_dataset in meta_data_list:
            if meta_data_dataset["contained_patients"]:
                new_contained_patients = list(
                    set(meta_data_dataset["contained_patients"] + meta_data["contained_patients"]))
                meta_data["contained_patients"] = new_contained_patients
        if detected_anomalies_df.empty or detected_anomalies_df is None:
            logger.info(f"Can not find any anomalies in the file {detected_anomalies_path}.")
            sys.exit(0)
        

        


        return detected_anomalies_df



    def _handle_anomalies_patient(self, anomaly_df: pd.DataFrame, relevant_data: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        """
            This function is used to clean the anomalies in the data. It is applied for each patient separately.

            Args:
                anomaly_df (pd.DataFrame): The DataFrame containing the detected anomalies.
                relevant_data (pd.DataFrame): The DataFrame containing the data to be fixed.
                original_data (pd.DataFrame): The complete DataFrame containing the original data.

            Returns:
                pd.DataFrame: The complete DataFrame for a patient with anomalies handled.
        """

        anomaly_df.drop(columns=["patient_id", "time"], inplace=True)
        if anomaly_df.empty or original_data.empty:
            logger.info("No data to fix. Exiting...")
            return pd.DataFrame(columns=original_data.columns)


        if self.handling_strategy == "delete_value":
            fixed_df = self._delete_value(anomaly_df, relevant_data)
        elif self.handling_strategy == "delete_than_impute":
            fixed_df = self._delete_than_impute(anomaly_df, relevant_data)
        elif self.handling_strategy == "delete_row_if_any_anomaly":
            fixed_df = self._delete_row_if_any_anomaly(anomaly_df, relevant_data)
        elif self.handling_strategy == "delete_row_if_many_anomalies":
            fixed_df = self._delete_row_if_many_anomalies(anomaly_df, relevant_data)
        elif self.handling_strategy == "use_prediction" and self.can_predict_value:
            fixed_df = self._use_prediction(anomaly_df, relevant_data)
        elif self.handling_strategy == "use_prediction" and not self.can_predict_value:
            raise NotImplementedError(f"Algorithm {self.type} does not support prediction as a handling strategy for anomalies.")
        elif self.handling_strategy == "use_prediction" and self.can_predict_value:
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown fixing strategy {self.handling_strategy}")
        finished_df = original_data
        finished_df.update(fixed_df)
        return finished_df

    @staticmethod
    def _fix_anomaly_df(anomaly_df: pd.DataFrame, starting_data: pd.DataFrame) -> pd.DataFrame:

        """
            This function processes the anomaly DataFrame to ensure it has an entry for each timestamp in the original patients data.
            If there is no entry for a given timestamp, it is assumed that no anomaly can be presented at this time and therefore the value is set to False

            Args:
                anomaly_df (pd.DataFrame): The DataFrame containing the detected anomalies.
                starting_data (pd.DataFrame): The DataFrame containing the data to be fixed.

            Returns:
                pd.DataFrame: The anomaly DataFrame having an entry for each timestamp and column in the original patients data.
        """

        anomaly_df_fixed = anomaly_df.copy(deep=True)
        all_timestamps = starting_data[["patient_id", "time"]].drop_duplicates(subset=["patient_id", "time"], keep="first")
        anomaly_df_fixed = pd.merge(all_timestamps, anomaly_df_fixed, how="outer", on=["patient_id", "time"])
        anomaly_df_fixed.fillna(False, inplace=True)
        anomaly_df_fixed = anomaly_df_fixed.sort_values(by=["time"], ascending=True).reset_index(drop=True)

        return anomaly_df_fixed

    def _calculate_anomaly_count(self, anomaly_df: pd.DataFrame, relevant_data: pd.DataFrame,
                                 anomaly_count_dict: dict = None) -> dict[str, dict[str, int]]:
        """
            This function counts the number of anomalies in each column of the DataFrame and the total amount of data present in each column.

            Args:
                anomaly_df (pd.DataFrame): The DataFrame containing the detected anomalies.
                relevant_data (pd.DataFrame): The DataFrame containing the values on which anomaly detection was done.
                anomaly_count_dict (dict): A dictionary to store the counts of anomalies and total data for each column.

            Returns:
                dict: A dictionary containing the counts of anomalies and total data for each column.
        """

        relevant_columns = [column for column in anomaly_df.columns if column not in self.columns_not_to_check]
        if self.patient_level_anomaly_detection:
            anomaly_count_dict = self._calculate_anomaly_count_per_step(anomaly_df, relevant_data, relevant_columns, anomaly_count_dict)
        else:
            anomaly_count_dict = self._calculate_anomaly_count_full(anomaly_df, relevant_data, relevant_columns)
        return anomaly_count_dict

    @staticmethod
    def _calculate_anomaly_count_per_step(anomaly_df: pd.DataFrame, relevant_data: pd.DataFrame, relevant_columns: list[str],
                                                anomaly_count_dict: dict = None) -> dict[str, dict[str, int]]:


        for column in relevant_columns:
            value_count = relevant_data[column].count()
            if column == "patient_id" or column == "time":
                continue
            anomaly_count = anomaly_df[column].sum()
            if column in anomaly_count_dict.keys():
                new_anomaly_count = anomaly_count_dict[column]["anomaly_count"] + anomaly_count
                new_value_count = anomaly_count_dict[column]["total_data"] + value_count
                anomaly_count_dict[column] = {"total_data": new_value_count,
                                              "anomaly_count": new_anomaly_count}
            else:
                anomaly_count_dict[column] = {"total_data": value_count,
                                              "anomaly_count": anomaly_count}
        return anomaly_count_dict




    def _handle_anomalies(self, detected_anomalies_df : pd.DataFrame, original_data: pd.DataFrame, save_data: bool =True, save_path: str = None, no_multi_processing: bool = False) -> pd.DataFrame:
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
            pass
        if not self.anomaly_counts:
            relevant_columns = [column for column in detected_anomalies_df.columns if column not in self.columns_not_to_check]
            anomaly_counts_dict = self._calculate_anomaly_count_full(detected_anomalies_df, relevant_data, relevant_columns)
            self.anomaly_counts = self.finalize_anomaly_counts_single(anomaly_counts_dict)

        original_data_df_list = [y for x,y in original_data_df.groupby("patient_id")]
        anomaly_df_list = [y for x,y in detected_anomalies_df.groupby("patient_id")]
        relevant_data_df_list = [y for x,y in relevant_data.groupby("patient_id")]
        if not no_multi_processing:

            with Pool(processes=self.max_processes) as pool:
                fixed_dfs = pool.starmap(self._handle_anomalies_patient,
                                         [(anomaly_df_list[i], relevant_data_df_list[i], original_data_df_list[i])
                                                                          for i in range(len(original_data_df_list))])
        else:
            fixed_dfs = []
            for i in range(len(original_data_df_list)):

                fixed_df = self._handle_anomalies_patient(anomaly_df_list[i], relevant_data_df_list[i], original_data_df_list[i])
                fixed_dfs.append(fixed_df)
        fixed_df = pd.concat(fixed_dfs, ignore_index=True).reset_index(drop=True)

        if save_data:
            fixed_data_path = os.path.join(save_path, f"fixed_data_{self.name}_{self.handling_strategy}_{self.fix_algorithm}.pkl")
            logger.info(f"Saving fixed data to {fixed_data_path}")
            self._save_file(fixed_df, fixed_data_path, True)
        return fixed_df

    @staticmethod
    def _delete_value(anomaly_df: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
            This function handles anomalies by deleting the values in the patients DataFrame that are marked as anomalies.
            It replaces the values with NaN.

            Args:
                anomaly_df (pd.DataFrame): The DataFrame containing the detected anomalies.
                dataframe (pd.DataFrame): The DataFrame containing the data to be fixed.

            Returns:
                pd.DataFrame: The processed DataFrame with anomalies handled.
        """
        dataframe = dataframe.mask(anomaly_df)
        return dataframe

    def _delete_than_impute(self, anomaly_df: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
            This function handles anomalies by deleting the values in the patients DataFrame that are marked as anomalies.
            After deleting the values, it applies a fixing algorithm to fill the missing values. Values that can not be fixed are set to NaN.

            Args:
                anomaly_df (pd.DataFrame): The DataFrame containing the detected anomalies.
                dataframe (pd.DataFrame): The DataFrame containing the data to be fixed.

            Returns:
                pd.DataFrame: The processed DataFrame with anomalies handled.
        """
        nan_mask = dataframe.isna()
        dataframe = dataframe.mask(anomaly_df)
        dataframe = self._fix_deleted(dataframe)
        dataframe = dataframe.mask(nan_mask)
        return dataframe

    @staticmethod
    def _delete_row_if_any_anomaly(anomaly_df: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
            This function handles anomalies by removing the whole row from the patients DataFrame if any of the values in the row are marked as anomalies.

            Args:
                anomaly_df (pd.DataFrame): The DataFrame containing the detected anomalies.
                dataframe (pd.DataFrame): The DataFrame containing the data to be fixed.

            Returns:
                pd.DataFrame: The processed DataFrame with anomalies handled.
        """
        dataframe = dataframe.mask(anomaly_df)
        index_to_drop = []
        for index, row in dataframe.iterrows():
            if row.isnull().any():
                index_to_drop.append(index)
        dataframe = dataframe.drop(index_to_drop).reset_index(drop=True)
        return dataframe

    def _delete_row_if_many_anomalies(self, anomaly_df: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
            This function handles anomalies by removing the whole row from the patients DataFrame if more than a certain percentage of the values in the row are marked as anomalies.
            The percentage is defined by the class variable anomaly_threshold.

            Args:
                anomaly_df (pd.DataFrame): The DataFrame containing the detected anomalies.
                dataframe (pd.DataFrame): The DataFrame containing the data to be fixed.

            Returns:
                pd.DataFrame: The processed DataFrame with anomalies handled.
        """
        n_columns = len(dataframe.columns)
        n_anomalies = anomaly_df.sum(axis=1)
        dataframe = dataframe.fillna(-100000)
        index_to_drop = []
        for i in range(len(dataframe)):
            if n_anomalies[i]/n_columns > self.anomaly_threshold:
                index_to_drop.append(i)
        dataframe = dataframe.drop(index_to_drop)
        dataframe = self._fix_deleted(dataframe)
        dataframe = dataframe.replace(-100000, pd.NA)
        return dataframe

    @staticmethod
    def _add_anomaly_score(anomaly_df: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
        for column in dataframe.columns:
            if column in anomaly_df.columns:
                dataframe[column + "_anomaly"] = anomaly_df[column]
        return dataframe

    def _fix_deleted(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if self.fix_algorithm == "forward":
            dataframe = dataframe.fillna(method="ffill")
        elif self.fix_algorithm == "backward":
            dataframe = dataframe.fillna(method="bfill")
        elif self.fix_algorithm == "interpolate":
            dataframe = dataframe.interpolate(method="linear", limit_direction="both")

        else:
            raise ValueError("Invalid fix_algorithm for Anomaly detection")
        return dataframe


    def create_meta_data(self) -> dict:
        meta_data_dict = {
            "name": self.name,
            "anomaly_detection_algorithm": None,
            "anomaly_handling_strategy": self.handling_strategy,
            "anomaly_fixing_algorithm": self.fix_algorithm,
            "columns_checked": self.columns_to_check,
            "anomaly_statistics":  self.anomaly_counts,
            "algorithm_specific_settings": None,
        }
        logger.info(self.anomaly_counts)
        if self.handling_strategy == "delete_row_if_many_anomalies":
            meta_data_dict["anomaly_threshold"] = self.anomaly_threshold
        else:
            meta_data_dict["anomaly_threshold"] = None
        if self.saved_anomaly_data:
            meta_data_dict["anomaly_data_dir"] = self.anomaly_data_dir

        return meta_data_dict




