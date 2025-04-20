from multiprocessing import Pool
from processing.processing_utils import prepare_multiprocessing

import pandas as pd


class BaseAnomalyDetector:

    def __init__(self, **kwargs):
        self.name = None
        self.type = None
        self.model = None
        self.columns_to_check = None
        self.database = None
        self.fix_algorithm = None
        self.handling_strategy = None
        self.anomaly_threshold = None
        self.max_processes = 4
        self.needs_full_data = False
        self.anomaly_counts = None
        # TODO add all static variables here
        self.columns_not_to_check = ["patient_id", "time", "ards", "chest-injury", "sepsis", "xray", "pneumonia", "pulmonary-edema", "hypervolemia", "heart-failure"]
        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)
        self.meta_data = None

    def run_handler(self, process_pool_data_list: list[pd.DataFrame], n_jobs: int, patients_per_process: int):

        if not self.needs_full_data:
            with Pool(processes=self.max_processes) as pool:
                anomaly_counts = None
                process_pool_data_list, anomaly_count_list = pool.starmap(self.run,
                                                      [(process_pool_data_list[i], i, n_jobs) for i in range(n_jobs)])
                for anomaly_count in anomaly_count_list:
                    if anomaly_counts is None:
                        for key, value in anomaly_count.items():
                            total_anomalies_name = key + "_total_anomalies"
                            percentage_anomalies_name = key + "_percentage_anomalies"
                            anomaly_counts = {total_anomalies_name: value,
                                              percentage_anomalies_name: None}

                    else:
                        for key in anomaly_count.keys():
                            total_anomalies_name = key + "_total_anomalies"
                            percentage_anomalies_name = key + "_percentage_anomalies"
                            if total_anomalies_name in anomaly_counts.keys():
                                anomaly_counts[total_anomalies_name] += anomaly_count[key]
                            else:
                                anomaly_counts[total_anomalies_name] = anomaly_count[key]
                            if not percentage_anomalies_name in anomaly_counts.keys():
                                anomaly_counts[percentage_anomalies_name] = None
        else:
            dataframe = pd.concat(process_pool_data_list, ignore_index=True).reset_index(drop=True)
            fixed_df, anomaly_counts = self.run(dataframe, 0, 1)
            process_pool_data_list, n_jobs = prepare_multiprocessing(fixed_df, patients_per_process)
        dataframe = pd.concat(process_pool_data_list, ignore_index=True).reset_index(drop=True)
        for key in anomaly_counts.keys():
            if key in dataframe.columns:
                total_anomalies_name = key + "_total_anomalies"
                if total_anomalies_name in anomaly_counts.keys():
                    percentage_anomalies_name = key + "_percentage_anomalies"
                    present_anomalies = dataframe[key].sum()
                    percentage = present_anomalies / len(dataframe)
                    anomaly_counts[percentage_anomalies_name] = percentage
        self.anomaly_counts = anomaly_counts

        return process_pool_data_list, n_jobs, dataframe

    def run(self,  dataframe_detection: pd.DataFrame, job_count: int, total_jobs: int) -> pd.DataFrame:
        """
            Runs the anomaly detection process from start to finish.

            Args:
                dataframe_detection (pd.DataFrame): The input DataFrame containing the data to be processed.
                job_count (int): The current job count (Irrelevant for approaches that need the full dataset).
                total_jobs (int): The total number of jobs (Irrelevant for approaches that need the full dataset).

            Returns:
                pd.DataFrame: The processed DataFrame with anomalies handled.

        """
        raise NotImplementedError()

    def _train_ad_model(self, data_training, data_validation, **kwargs):

        raise NotImplementedError()

    def _predict(self, dataframe: pd.DataFrame, **kwargs) -> dict:
        raise NotImplementedError()

    def _predict_proba(self):
        raise NotImplementedError()

    def _prepare_data(self, dataframe: pd.DataFrame) -> dict:
        raise NotImplementedError()

    def _handle_anomalies(self, anomalies: dict, anomalous_data : pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    @staticmethod
    def _delete_value(anomaly_df: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.mask(anomaly_df)
        return dataframe

    def _delete_than_impute(self, anomaly_df: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.fillna(-100000)
        dataframe = dataframe.mask(anomaly_df)
        dataframe = self._fix_deleted(dataframe)
        dataframe = dataframe.replace(-100000, pd.NA)
        return dataframe

    @staticmethod
    def _delete_row_if_any_anomaly(anomaly_df: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:

        dataframe = dataframe.mask(anomaly_df)
        index_to_drop = []
        for index, row in dataframe.iterrows():
            if row.isnull().any():
                index_to_drop.append(index)
        dataframe = dataframe.drop(index_to_drop)
        return dataframe

    def _delete_row_if_many_anomalies(self, anomaly_df: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:

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

    def _use_prediction(self, prediction: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

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
        if self.handling_strategy == "delete_row_if_many_anomalies":
            meta_data_dict["anomaly_threshold"] = self.anomaly_threshold
        else:
            meta_data_dict["anomaly_threshold"] = None
        return meta_data_dict




