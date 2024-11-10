import numpy as np
import pandas as pd

class DataImputator:
    def __init__(self, config) -> None:
        self.available_imputation_methods = ["forward", "backfill", "mean", "linear_interpolation"]
        self.set_imputation_method(config["default_imputation_method"])
        self.imputation_method = None
        self.params_to_impute = config["params_to_impute"]
        self.default_imputation_method = config["default_imputation_method"]
        self.impute_empty_cells = config["impute_empty_cells"]
        self.binary_variables = ["ards", "heart-failure", "hypervolemia", "mech-vent", "pneumonia", "xray", "sepsis",
                                 "chest-injury"]

    def impute_missing_data(self, dataframe: pd.DataFrame, job_number: int, total_job_count: int) -> pd.DataFrame:
        print("Start imputation for job " + str(job_number) + f" of {total_job_count} jobs...")
        columns = dataframe.columns
        for param in self.params_to_impute:
            param = param.split(", ")
            if len(param) > 1:
                self.set_imputation_method(param[1])
            else:
                self.set_imputation_method(self.default_imputation_method)
            param = param[0]
            if param in self.binary_variables and (self.imputation_method == "mean" or self.imputation_method == "linear_interpolation"):
                raise RuntimeError("Please use only forward or backward fill as imputation method for binary variables!")
            for series_name, series in dataframe.items():
                if series_name == param or param == "all":
                    if self.imputation_method == "forward": 
                        dataframe[series_name] = dataframe.groupby("patient_id")[series_name].ffill()
                        dataframe[series_name] = dataframe.groupby("patient_id")[series_name].bfill() #to ensure that there will not remain empty marginal values
                    if self.imputation_method == "backfill": 
                        dataframe[series_name] = dataframe.groupby("patient_id")[series_name].bfill()
                        dataframe[series_name] = dataframe.groupby("patient_id")[series_name].ffill() #to ensure that there will not remain empty marginal values
                    if self.imputation_method == "mean":
                        temp_dataframe = pd.DataFrame(columns=[series_name])
                        for name, frame in dataframe.groupby("patient_id")[series_name]:
                            temp_dataframe = pd.concat(
                                [temp_dataframe, pd.DataFrame(frame.fillna(value=frame.mean(skipna=True)))],
                                ignore_index=True)
                        dataframe[series_name] = temp_dataframe
                    if self.imputation_method == "linear_interpolation":
                        temp_dataframe = pd.DataFrame(columns=[series_name])
                        for name, frame in dataframe.groupby("patient_id")[series_name]:
                            temp_dataframe = pd.concat([temp_dataframe, pd.DataFrame(
                                frame.interpolate(method="linear", limit_direction="both"))], ignore_index=True)
                        dataframe[series_name] = temp_dataframe

        dataframe.dropna(subset=['ards'], inplace=True, ignore_index=True, how="any", axis=0)
        if not self.impute_empty_cells:
            dataframe.dropna(how="all", axis=1, ignore_index=True, inplace=True)
            dataframe.dropna(how="any", axis=0, ignore_index=True, inplace=True)
        else:
            dataframe.fillna(value=-100000, axis=1, inplace=True)
        if len(dataframe.index) ==0:
            dataframe = pd.DataFrame(columns=columns)
        print("Finished imputation for job " + str(job_number) + f" of {total_job_count} jobs...")
        return dataframe
    
    def set_imputation_method(self, method):
        if method in self.available_imputation_methods:
            self.imputation_method = method
        else:
            raise RuntimeError(
                "Imputation method " + method + " not available/implemented! Currently available methods are " + str(
                    self.available_imputation_methods))

    @staticmethod
    def impute_rows(dataframe, target_start_time, target_end_time):
        while dataframe["time"][dataframe.index[0]] > target_start_time:
            empty_data = pd.DataFrame({col: [np.nan for _ in range(1)] for col in dataframe.columns})
            dataframe = pd.concat([empty_data, dataframe], ignore_index=True)
            dataframe["time"].interpolate(method="spline", order=1, inplace=True, limit_direction="both")
            dataframe.fillna(value=-100000, axis=1, inplace=True)
        while dataframe["time"][dataframe.index[len(dataframe.index) - 1]] < target_end_time:
            empty_data = pd.DataFrame({col: [np.nan for _ in range(1)] for col in dataframe.columns})
            dataframe = pd.concat([dataframe, empty_data], ignore_index=True)
            dataframe["time"].interpolate(method="spline", order=1, inplace=True, limit_direction="both")
            dataframe.fillna(value=-100000, axis=1, inplace=True)
        return dataframe
            