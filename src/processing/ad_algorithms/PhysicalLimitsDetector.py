import logging

import pandas as pd

from processing.ad_algorithms.torch_utils import check_directory
from processing.datasets_metadata import AnomalyDetectionMetaData

from processing.ad_algorithms.BaseAnomalyDetector import BaseAnomalyDetector
from processing.ad_algorithms.configs import physical_limits_database_dict


logger = logging.getLogger(__name__)
class PhysicalLimitsDetector(BaseAnomalyDetector):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.anomaly_data_dir = kwargs.get("anomaly_data_dir", "../Data/AnomalyData/PhysicalLimits/")
        check_directory(self.anomaly_data_dir)
        self.type = "PhysicalLimits"
        self.model = None
        self.physical_limits_dict = None
        self.needs_full_data = False

        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)
        self.physical_limits_dict = physical_limits_database_dict[self.database]

    def run(self, dataframe_detection: pd.DataFrame, job_count: int, total_jobs: int) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:

        relevant_data = self._prepare_data(dataframe_detection)["dataframe"]
        anomaly_dict = self._predict(relevant_data)
        anomaly_df =  anomaly_dict["anomaly_df"]
        self._save_anomaly_df(anomaly_df)


        fixed_df = self._handle_anomalies(anomaly_df, dataframe_detection)
        return fixed_df, anomaly_dict["anomaly_count"]





    def _prepare_data(self, dataframe: pd.DataFrame, save_data: bool =False, overwrite: bool = True) -> dict:
        dataframe = dataframe[self.columns_to_check]
        return_dict = {"dataframe": dataframe}
        if save_data:
            first_patient, last_patient = self._get_first_and_last_patient_id_for_name(dataframe)
            save_path = f"{self.prepared_data_dir}/patient_{first_patient}_to_{last_patient}.pkl"
            self._save_file(return_dict, save_path, overwrite)
        return return_dict

    def _predict(self, dataframe: pd.DataFrame, **kwargs) -> dict:
        anomaly_dict = {}
        for column in dataframe.columns:
            if self.columns_to_check == [] or column in self.columns_to_check:

                anomaly_dict[column] = []
                min_value = self.physical_limits_dict[column]["min"]
                max_value = self.physical_limits_dict[column]["max"]
                column_present = column in self.physical_limits_dict.keys()
                for index, row in dataframe.iterrows():
                    if row[column] is None or not column_present:
                        anomaly_dict[column].append(False)
                    elif row[column] >= min_value or row[column] <= max_value:
                        anomaly_dict[column].append(True)
                    else:
                        anomaly_dict[column].append(False)
        anomaly_df = pd.DataFrame.from_dict(anomaly_dict)
        anomaly_df["patient_id"] = dataframe["patient_id"]
        anomaly_df["timestamp"] = dataframe["timestamp"]
        anomaly_count_dict = self._calculate_anomaly_count(anomaly_df, dataframe)
        result_dict = {
            "anomaly_df": anomaly_df,
            "anomaly_count": anomaly_count_dict}
        return result_dict



    def create_meta_data(self):
        meta_data_dict = super().create_meta_data()
        meta_data_dict["anomaly_detection_algorithm"] = self.type
        meta_data_dict["algorithm_specific_settings"] = {"physical_limits_dict": self.physical_limits_dict}
        return AnomalyDetectionMetaData(**meta_data_dict)

