import numpy as np
import pandas as pd

from processing.ad_algorithms.AnomalyDetector import AnomalyDetector
from processing.ad_algorithms.configs import physical_limits_database_dict

class PhysicalLimitsDetector(AnomalyDetector):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "PhysicalLimits"
        self.model = None
        self.physical_limits_dict = None
        self.needs_full_data = False

        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)
        self.physical_limits_dict = physical_limits_database_dict[self.database]

    def run(self, dataframe_detection: pd.DataFrame, job_count: int, total_jobs: int):

        relevant_data = self._prepare_data(dataframe_detection)["dataframe"]
        anomaly_dict = self._predict(relevant_data)

        fixed_df = self.handle_anomalies(anomaly_dict, relevant_data, dataframe_detection)
        return fixed_df


    def handle_anomalies(self, anomaly_dict: dict, relevant_data: pd.DataFrame, original_data: pd.DataFrame):
        anomaly_df = pd.DataFrame.from_dict(anomaly_dict)
        if self.handling_strategy == "delete_value":
            fixed_df = self._delete_value(anomaly_df, original_data)
        elif self.handling_strategy == "delete_than_impute":
            fixed_df = self._delete_than_impute(anomaly_df, relevant_data)
        elif self.handling_strategy == "delete_row_if_any_anomaly":
            fixed_df = self._delete_row_if_any_anomaly(anomaly_df, relevant_data)
        elif self.handling_strategy == "delete_row_if_many_anomalies":
            fixed_df = self._delete_row_if_many_anomalies(anomaly_df, relevant_data)
        elif self.handling_strategy == "use_prediction":
            raise ValueError("Fixing strategy 'use_prediction' is not implemented for PhysicalLimitsDetector")
        else:
            raise ValueError("Unknown fixing strategy")
        finished_df = original_data
        finished_df.update(fixed_df)
        return finished_df


    def _prepare_data(self, dataframe: pd.DataFrame) -> dict:
        dataframe = dataframe[self.columns_to_check]
        return_dict = {"dataframe": dataframe}
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
        return anomaly_dict

