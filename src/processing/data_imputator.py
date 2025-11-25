import numpy as np
import pandas as pd
import logging

from processing.datasets_metadata import ImputationMetaData

logger = logging.getLogger(__name__)

class DataImputator:
    def __init__(self, config) -> None:
        self.available_imputation_methods = ["forward", "backfill", "mean", "linear_interpolation"]
        self.set_imputation_method(config["default_imputation_method"])
        self.imputation_method = None
        self.params_to_impute = config["params_to_impute"]
        self.default_imputation_active = False
        self.param_imputation_method = {}
        for param in self.params_to_impute:
            param = param.split(", ")
            if len(param) > 1:
                self.param_imputation_method[param[0]] = param[1]
            else:
                self.param_imputation_method[param[0]] = config["default_imputation_method"]

        self.default_imputation_method = config["default_imputation_method"]
        self.impute_empty_cells = config["impute_empty_cells"]
        self.binary_variables = ["ards", "heart-failure", "hypervolemia", "mech-vent", "pneumonia", "xray", "sepsis",
                                 "chest-injury"]
        self.meta_data = None
        self.total_imputed_values = 0

    @staticmethod
    def _fast_fillna_numeric(df: pd.DataFrame, fill_value: float = -100000) -> None:
        """
        Fast in-place fill of NaNs for numeric columns using numpy arrays.
        This avoids the overhead of pandas.fillna across the whole frame when only
        numeric columns need the sentinel value.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return
        arr = df[numeric_cols].to_numpy(copy=False)
        # create mask and fill (np.isnan works fast on numeric numpy arrays)
        mask = np.isnan(arr)
        if mask.any():
            arr[mask] = fill_value
            # assign back (works because arr is a view when copy=False and dtypes compatible)
            df[numeric_cols] = arr

    def impute_missing_data(self, dataframe: pd.DataFrame, job_number: int, total_job_count: int) -> pd.DataFrame:
        """
        Imputes missing data in the DataFrame according to configured imputation methods.
        
        Args:
            dataframe: Input DataFrame with missing values
            job_number: Current job number for logging
            total_job_count: Total number of jobs for logging
            
        Returns:
            DataFrame with imputed values
        """
        logger.info(f"Start imputation for job {job_number} of {total_job_count} jobs...")
        
        # Track rows before imputation for statistics
        rows_before = len(dataframe)
        unique_patients_before = dataframe["patient_id"].nunique() if not dataframe.empty else 0
        logger.debug(f"Job {job_number}: Input has {rows_before} rows, {unique_patients_before} unique patients")
        
        columns = dataframe.columns
        job_imputed_values = 0
        processed_patients = 0
        checkpoint_interval = 1000

        for column in columns:
            if column in self.param_imputation_method.keys():
                self.imputation_method = self.param_imputation_method[column]
            elif "all" in self.param_imputation_method.keys():
                self.imputation_method = self.param_imputation_method["all"]
            else:
                continue

            if column in self.binary_variables and self.imputation_method not in ["forward", "backfill"]:
                logger.error(f"Invalid imputation method '{self.imputation_method}' for binary variable '{column}'")
                raise RuntimeError("Please use only forward or backward fill as imputation method for binary variables!")

            logger.debug(f"Job {job_number}: Imputing column '{column}' using method '{self.imputation_method}'")
            
            # Count NaN values before imputation for this column
            nan_count_before = dataframe[column].isna().sum()
            
            if self.imputation_method == "forward":
                dataframe[column] = dataframe.groupby("patient_id")[column].ffill()
                dataframe[column] = dataframe.groupby("patient_id")[column].bfill()
            elif self.imputation_method == "backfill":
                dataframe[column] = dataframe.groupby("patient_id")[column].bfill()
                dataframe[column] = dataframe.groupby("patient_id")[column].ffill()
            elif self.imputation_method == "mean":
                for name, frame in dataframe.groupby("patient_id")[column]:
                    processed_patients += 1
                    if processed_patients % checkpoint_interval == 0:
                        logger.debug(f"Job {job_number}: Processed {processed_patients} patients (mean imputation)...")
                    dataframe.loc[frame.index, column] = frame.fillna(value=frame.mean(skipna=True))
                processed_patients = 0
            elif self.imputation_method == "linear_interpolation":
                for name, frame in dataframe.groupby("patient_id")[column]:
                    processed_patients += 1
                    if processed_patients % checkpoint_interval == 0:
                        logger.debug(f"Job {job_number}: Processed {processed_patients} patients (linear interpolation)...")
                    dataframe.loc[frame.index, column] = frame.interpolate(method="linear", limit_direction="both")
                processed_patients = 0
            
            # Count imputed values for this column
            nan_count_after = dataframe[column].isna().sum()
            values_imputed = nan_count_before - nan_count_after
            job_imputed_values += values_imputed

        # Handle rows with missing ARDS values
        rows_with_ards_na = dataframe["ards"].isna().sum()
        dataframe.dropna(subset=['ards'], inplace=True, how="any", axis=0)
        logger.debug(f"Job {job_number}: Dropped {rows_with_ards_na} rows with missing 'ards' values")

        dataframe.reset_index(drop=True, inplace=True)
        
        if not self.impute_empty_cells:
            logger.debug(f"Job {job_number}: Dropping rows and columns with remaining NaN values...")
            dataframe.dropna(how="all", axis=1, inplace=True)
            dataframe.reset_index(drop=True, inplace=True)
            dataframe.dropna(how="any", axis=0, inplace=True)
            dataframe.reset_index(drop=True, inplace=True)
        else:
            logger.debug(f"Job {job_number}: Filling remaining empty cells with -100000 (numeric columns only for speed)...")
            # Fast path: only fill numeric columns with the sentinel using numpy to speed up large frames
            try:
                self._fast_fillna_numeric(dataframe, fill_value=-100000)
                # If you also want to force-fill non-numeric columns (may coerce dtypes),
                # do it with pandas but only if necessary and expect it to be slower:
                non_numeric_cols = dataframe.columns.difference(dataframe.select_dtypes(include=[np.number]).columns)
                if len(non_numeric_cols) > 0:
                    # Use pandas.fillna on the subset (still faster than filling whole df repeatedly)
                    dataframe[non_numeric_cols] = dataframe[non_numeric_cols].fillna(value=-100000)
            except Exception as e:
                logger.warning(f"Fast numeric fill failed, falling back to pandas.fillna: {e}")
                dataframe.fillna(value=-100000, inplace=True)
        
        if len(dataframe.index) == 0:
            logger.warning(f"Job {job_number}: All rows were dropped during imputation, returning empty DataFrame")
            dataframe = pd.DataFrame(columns=columns)
        
        rows_after = len(dataframe)
        unique_patients_after = dataframe["patient_id"].nunique() if not dataframe.empty else 0
        logger.info(f"Finished imputation for job {job_number}: {rows_after} rows remaining, "
                   f"{unique_patients_after} patients, {job_imputed_values} values imputed")
        
        # Accumulate total imputed values across all jobs
        self.total_imputed_values += job_imputed_values
        
        return dataframe

    def create_meta_data(self):
        logger.debug(f"Creating imputation metadata. Total imputed values across all jobs: {self.total_imputed_values}")
        self.meta_data = ImputationMetaData(impute_empty_cells=self.impute_empty_cells, 
                                          imputation_parameter_algorithm_dict=self.param_imputation_method)
        logger.info(f"Imputation metadata created. Total values imputed: {self.total_imputed_values}")

    def set_imputation_method(self, method):
        if method in self.available_imputation_methods:
            self.imputation_method = method
            logger.debug(f"Imputation method set to: {method}")
        else:
            logger.error(f"Imputation method '{method}' not available. Available methods: {self.available_imputation_methods}")
            raise RuntimeError(
                "Imputation method " + method + " not available/implemented! Currently available methods are " + str(
                    self.available_imputation_methods))

    @staticmethod
    def impute_rows(dataframe, target_start_time, target_end_time):
        """
        Imputes rows to fill time gaps at the beginning and end of a patient's timeseries.
        
        Args:
            dataframe: Patient timeseries DataFrame
            target_start_time: Target start time
            target_end_time: Target end time
            
        Returns:
            DataFrame with imputed rows
        """
        # avoid repeated small concats: compute how many rows to add at once
        if len(dataframe) == 0:
            return dataframe

        # start
        first_time = dataframe["time"].iat[0]
        if first_time > target_start_time:
            # estimate reasonable number of rows to add using median delta (fallback to 1)
            if len(dataframe) > 1:
                deltas = dataframe["time"].diff().dropna()
                median_delta = deltas.median() if not deltas.empty else 1
                if median_delta <= 0 or np.isnan(median_delta):
                    median_delta = 1
                n_add = int(np.ceil((first_time - target_start_time) / median_delta))
            else:
                n_add = 1
            if n_add > 0:
                empty_block = pd.DataFrame(np.nan, index=range(n_add), columns=dataframe.columns)
                dataframe = pd.concat([empty_block, dataframe], ignore_index=True)

        # end
        last_time = dataframe["time"].iat[len(dataframe.index) - 1]
        if last_time < target_end_time:
            if len(dataframe) > 1:
                deltas = dataframe["time"].diff().dropna()
                median_delta = deltas.median() if not deltas.empty else 1
                if median_delta <= 0 or np.isnan(median_delta):
                    median_delta = 1
                n_add_end = int(np.ceil((target_end_time - last_time) / median_delta))
            else:
                n_add_end = 1
            if n_add_end > 0:
                empty_block = pd.DataFrame(np.nan, index=range(n_add_end), columns=dataframe.columns)
                dataframe = pd.concat([dataframe, empty_block], ignore_index=True)

        # single interpolation call (less overhead)
        dataframe["time"].interpolate(method="spline", order=1, inplace=True, limit_direction="both")
        # fast numeric fill of remaining NaNs
        DataImputator._fast_fillna_numeric(dataframe, fill_value=-100000)
        # for any non-numeric columns that still have NaN, fallback to pandas fill
        non_numeric_cols = dataframe.columns.difference(dataframe.select_dtypes(include=[np.number]).columns)
        if len(non_numeric_cols) > 0:
            dataframe[non_numeric_cols] = dataframe[non_numeric_cols].fillna(value=-100000)
        return dataframe
