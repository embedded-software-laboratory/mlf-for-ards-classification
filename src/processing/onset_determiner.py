import pandas as pd
import logging
from processing.datasets_metadata import OnsetDetectionMetaData

logger = logging.getLogger(__name__)

class OnsetDeterminer:  # Class to determine the ARDS onset in a given dataset
    def __init__(self, config, data_imputator) -> None:
        """
        Initializes the OnsetDeterminer with configuration settings and a data imputator.
        
        Args:
            config: Configuration dictionary containing detection and return settings
            data_imputator: An instance of a data imputator for handling missing data
        """
        logger.info("Initializing OnsetDeterminer...")
        self.detection_rule = "first_horovitz"
        # The available methods to determine the ards onset in the data (the peep must be at least 5 for all horovitz values)
        # 4h / 12h / 24h: The first timestamp at which the horovitz is under 300 for the next 4 / 12 / 24 hours
        # 4h_50: The first timestamp at which the horovitz is under 300 for 50% of the values for the next 4 hours
        # lowest_horovitz: the lowest horovitz in the given data
        # first_horovitz: the first horovitz which is under 300
        self.available_detection_rules = ["4h", "4h_50", "12h", "24h", "lowest_horovitz", "first_horovitz"]
        self.return_rule = "datapoint"
        self.available_return_rules = ["datapoint", "data_series_as_series", "data_series_as_point"]
        self.return_series_start = None  # In seconds, relative to the determined ARDS onset
        self.return_series_end = None  # In seconds, relative to the determined ARDS onset
        self.set_detection_rule(config["detection_rule"])
        self.set_return_rule(config["return_rule"], config["series_start_point"], config["series_end_point"])
        self.remove_ards_patients_without_onset = config["remove_ards_patients_without_onset"]
        self.impute_missing_rows = config["impute_missing_rows"]
        self.data_imputator = data_imputator
        self.update_ards_values = config["update_ards_values"]
        self.meta_data = None
        logger.info("OnsetDeterminer initialized successfully.")

    def create_meta_data(self):
        """
        Creates metadata for the onset detection process.
        """
        logger.info("Creating metadata for onset detection...")
        return_series_start = self.return_series_start
        return_series_end = self.return_series_end
        update_ards_values = self.update_ards_values
        impute_missing_rows = self.impute_missing_rows

        self.meta_data = OnsetDetectionMetaData(
            onset_detection_algorithm=self.detection_rule,
            onset_detection_return_type=self.return_rule,
            series_begin=return_series_start,
            series_end=return_series_end,
            remove_ards_patients_without_onset=self.remove_ards_patients_without_onset,
            update_ards_diagnose=update_ards_values,
            fill_cells_out_of_patient_stay=impute_missing_rows
        )
        logger.info("Metadata created successfully.")

    def set_detection_rule(self, rule):
        """
        Sets the detection rule for ARDS onset detection.
        
        Args:
            rule: The detection rule to set
            
        Raises:
            RuntimeError: If the rule is not available
        """
        logger.info(f"Setting detection rule to: {rule}")
        if rule not in self.available_detection_rules:
            error_msg = f"Detection rule {rule} is not implemented. Available rules are: {self.available_detection_rules}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        self.detection_rule = rule
        logger.info("Detection rule set successfully.")

    def set_return_rule(self, rule, series_start=None, series_end=None):
        """
        Sets the return rule for the output of the onset detection.
        
        Args:
            rule: The return rule to set
            series_start: Start point for the return series
            series_end: End point for the return series
            
        Raises:
            RuntimeError: If the rule is not available or series length is not defined
            ValueError: If the end of the series is before the start
        """
        logger.info(f"Setting return rule to: {rule}")
        if rule not in self.available_return_rules:
            error_msg = f"Return rule {rule} is not implemented. Available rules are: {self.available_return_rules}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        if (rule in ["data_series_as_series", "data_series_as_point"]) and (series_start is None or series_end is None):
            error_msg = "Series length not defined but needed for returning a data series."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        if series_end < series_start:
            error_msg = "The end of the series must be after the beginning of the series."
            logger.error(error_msg)
            raise ValueError(error_msg)
        self.return_rule = rule
        self.return_series_start = series_start
        self.return_series_end = series_end
        logger.info("Return rule set successfully.")

    def determine_ards_onset(self, dataframe, job_number: int, total_job_count: int):
        """
        Determines the ARDS onset for each patient in the provided DataFrame.
        
        Args:
            dataframe: DataFrame containing patient data
            job_number: Current job number for logging
            total_job_count: Total number of jobs for logging
            
        Returns:
            DataFrame containing the results of the onset detection
        """
        logger.info(f"Start onset detection for job {job_number} of {total_job_count} jobs...")
        logger.debug(f"Input dataframe shape: {dataframe.shape}")
        logger.debug(f"Input dataframe columns: {dataframe.columns.tolist()}")
        
        # Validate critical columns
        logger.debug("Validating critical columns...")
        
        # Validate horovitz column
        if "horovitz" not in dataframe.columns:
            logger.error("Column 'horovitz' not found in dataframe")
            raise ValueError("Column 'horovitz' is required for ARDS onset detection")
        
        # Validate peep column
        if "peep" not in dataframe.columns:
            logger.error("Column 'peep' not found in dataframe")
            raise ValueError("Column 'peep' is required for ARDS onset detection")
        
        # Validate time column
        if "time" not in dataframe.columns:
            logger.error("Column 'time' not found in dataframe")
            raise ValueError("Column 'time' is required for ARDS onset detection")
        
        # Validate patient_id column
        if "patient_id" not in dataframe.columns:
            logger.error("Column 'patient_id' not found in dataframe")
            raise ValueError("Column 'patient_id' is required for ARDS onset detection")
        
        # Validate ards column
        if "ards" not in dataframe.columns:
            logger.error("Column 'ards' not found in dataframe")
            raise ValueError("Column 'ards' is required for ARDS onset detection")
        
        logger.info("All critical columns validated successfully.")
        
        return_dataframe = pd.DataFrame()
        unique_patients = set(dataframe["patient_id"].tolist())
        logger.debug(f"Processing {len(unique_patients)} unique patients")
        
        for patient_count, patientid in enumerate(unique_patients, 1):
            logger.debug(f"Processing patient {patient_count}/{len(unique_patients)}: ID={patientid}")
            patient_mask = dataframe["patient_id"] == patientid
            patient_data = dataframe[patient_mask]
            logger.debug(f"  Patient data shape: {patient_data.shape}")
            
            if self.detection_rule == "first_horovitz":
                logger.debug(f"  Using detection rule: first_horovitz")
                horovitz_index = -1
                for index in patient_data.index:
                    horovitz_val = dataframe.loc[index, "horovitz"]
                    peep_val = dataframe.loc[index, "peep"]
                    logger.debug(f"    Index {index}: horovitz={horovitz_val}, peep={peep_val}")
                    
                    if horovitz_val < 300 and peep_val >= 5:
                        horovitz_index = index
                        logger.debug(f"    Found onset at index {index}")
                        break
                
                if horovitz_index != -1:
                    return_dataframe = self.add_return_data(return_dataframe, dataframe, horovitz_index)
                else:
                    logger.debug(f"  No valid onset found via first_horovitz. Checking fallback conditions...")
                    ards_value = dataframe.loc[patient_data.index[0], "ards"]
                    if (ards_value == 0 or not self.remove_ards_patients_without_onset):
                        horovitz_index = patient_data["horovitz"].idxmin()
                        logger.debug(f"    Using fallback: minimum horovitz at index {horovitz_index}")
                        return_dataframe = self.add_return_data(return_dataframe, dataframe, horovitz_index)
                    else:
                        logger.debug(f"    Patient has ARDS=1 and remove_ards_patients_without_onset=True. Skipping patient.")

            elif self.detection_rule == "lowest_horovitz":
                logger.debug(f"  Using detection rule: lowest_horovitz")
                try:
                    horovitz_index = patient_data["horovitz"].idxmin()
                    horovitz_value = dataframe.loc[horovitz_index, "horovitz"]
                    logger.debug(f"  Minimum horovitz value: {horovitz_value} at index {horovitz_index}")
                    return_dataframe = self.add_return_data(return_dataframe, dataframe, horovitz_index)
                except Exception as e:
                    logger.error(f"  Error finding lowest horovitz for patient {patientid}: {e}")
                    raise

            elif self.detection_rule in ["4h", "12h", "24h"]:
                logger.debug(f"  Using detection rule: {self.detection_rule}")
                period_seconds = {"4h": 14400, "12h": 43200, "24h": 86400}[self.detection_rule]
                logger.debug(f"  Period length: {period_seconds} seconds")
                return_dataframe = self.get_low_horovitz_period_start(dataframe, period_seconds,
                                                                      patientid, return_dataframe)

            elif self.detection_rule == "4h_50":
                logger.debug(f"  Using detection rule: 4h_50")
                horovitz_index = -1
                start_index_of_closest_series = -1
                horovitz_percentage_of_closest_series = 0
                
                for index in patient_data.index:
                    horovitz_val = dataframe.loc[index, "horovitz"]
                    peep_val = dataframe.loc[index, "peep"]
                    
                    if horovitz_val < 300 and peep_val >= 5:
                        horovitz_index = index
                        start_timestamp = dataframe.loc[index, "time"]
                        second_index = index
                        number_of_low_horovitz = 1
                        total_number_of_rows = 1
                        fulfilled = True
                        
                        while (second_index < len(dataframe) and
                               dataframe.loc[second_index, "patient_id"] == patientid and
                               dataframe.loc[second_index, "time"] - start_timestamp <= 14400):
                            total_number_of_rows += 1
                            if dataframe.loc[second_index, "horovitz"] < 300:
                                number_of_low_horovitz += 1
                            second_index += 1
                        
                        percentage = number_of_low_horovitz / total_number_of_rows
                        logger.debug(f"    Index {index}: low horovitz percentage = {percentage:.2%}")
                        
                        if percentage <= 0.5:
                            fulfilled = False
                            if percentage > horovitz_percentage_of_closest_series:
                                horovitz_percentage_of_closest_series = percentage
                                start_index_of_closest_series = horovitz_index
                        
                        if fulfilled:
                            logger.debug(f"    Found valid 4h_50 onset at index {horovitz_index}")
                            break
                        else:
                            horovitz_index = -1
                
                if horovitz_index != -1:
                    return_dataframe = self.add_return_data(return_dataframe, dataframe, horovitz_index)
                elif start_index_of_closest_series != -1:
                    ards_value = dataframe.loc[patient_data.index[0], "ards"]
                    if (ards_value == 0 or not self.remove_ards_patients_without_onset):
                        logger.debug(f"    Using closest series as fallback at index {start_index_of_closest_series}")
                        return_dataframe = self.add_return_data(return_dataframe, dataframe, start_index_of_closest_series)
                else:
                    ards_value = dataframe.loc[patient_data.index[0], "ards"]
                    if (ards_value == 0 or not self.remove_ards_patients_without_onset):
                        horovitz_index = patient_data["horovitz"].idxmin()
                        logger.debug(f"    Using minimum horovitz as final fallback at index {horovitz_index}")
                        return_dataframe = self.add_return_data(return_dataframe, dataframe, horovitz_index)

        if len(return_dataframe.index) == 0:
            logger.warning(f"No onset data found for any patients in job {job_number}")
            return_dataframe = pd.DataFrame(columns=dataframe.columns)
        else:
            logger.info(f"Onset detection complete for job {job_number}: {len(return_dataframe)} rows detected")
        
        logger.info(f"Finished onset detection for job {job_number} of {total_job_count} jobs...")
        return return_dataframe

    def add_return_data(self, return_dataframe, dataframe, index):
        """
        Adds the detected onset data to the return DataFrame based on the return rule.
        
        Args:
            return_dataframe: DataFrame to which the data will be added
            dataframe: Original DataFrame containing patient data
            index: Index of the detected onset data
            
        Returns:
            Updated return DataFrame
        """
        logger.debug(f"Adding return data for index: {index} with return rule: {self.return_rule}")
        if self.return_rule == "datapoint":
            return_dataframe = pd.concat([return_dataframe, dataframe.iloc[[index]]], axis=0)
            logger.debug("Return data added as a single data point.")
            return return_dataframe

        elif self.return_rule == "data_series_as_series":
            start_index, end_index = self.get_sub_series_indices(dataframe, index)
            new_dataframe = dataframe.iloc[start_index:end_index]
            if self.impute_missing_rows:
                threshold_timestamp = dataframe["time"][index]
                new_dataframe = self.data_imputator.impute_rows(new_dataframe,
                                                                threshold_timestamp + self.return_series_start,
                                                                threshold_timestamp + self.return_series_end)
            if self.update_ards_values and dataframe["ards"][index] == 1:
                for i in new_dataframe.index:
                    new_dataframe["ards"][i] = 0 if i < index else 1
            return_dataframe = pd.concat([return_dataframe, new_dataframe], axis=0, ignore_index=True)
            logger.debug("Return data added as a series.")
            return return_dataframe

        elif self.return_rule == "data_series_as_point":
            start_index, end_index = self.get_sub_series_indices(dataframe, index)
            temp_dataframe = pd.DataFrame(
                [[dataframe["patient_id"][index], dataframe["ards"][index], dataframe["time"][index]]],
                columns=["patient_id", "ards", "time"])
            new_dataframe = dataframe.iloc[start_index:end_index]
            if self.impute_missing_rows:
                threshold_timestamp = dataframe["time"][index]
                new_dataframe = self.data_imputator.impute_rows(new_dataframe,
                                                                threshold_timestamp + self.return_series_start,
                                                                threshold_timestamp + self.return_series_end)
            new_dataframe = new_dataframe.drop(["patient_id", "ards", "time"], axis=1)
            counter = 0
            for i in range(start_index, end_index):
                column_names = [columnname + str(counter) for columnname in new_dataframe.columns]
                values = new_dataframe.loc[i].to_list()
                temp_dataframe2 = pd.DataFrame([values], columns=column_names)
                temp_dataframe = pd.concat([temp_dataframe, temp_dataframe2], axis=1)
                counter += 1
            return_dataframe = pd.concat([return_dataframe, temp_dataframe], axis=0)
            logger.debug("Return data added as a series of points.")
            return return_dataframe

    def get_low_horovitz_period_start(self, dataframe, period_length, patientid, return_dataframe):
        """
        Finds the start of a low Horovitz period for a given patient.
        
        Args:
            dataframe: DataFrame containing patient data
            period_length: Length of the period to check for low Horovitz values
            patientid: ID of the patient to check
            return_dataframe: DataFrame to which the data will be added
            
        Returns:
            Updated return DataFrame
        """
        logger.debug(f"Getting low Horovitz period start for patient ID: {patientid}")
        horovitz_index = -1
        start_timestamp = 0
        series_length = 0
        series_start = -1
        for index in dataframe[dataframe["patient_id"] == patientid].index:
            if dataframe["horovitz"][index] < 300 and dataframe["peep"][index] >= 5:
                horovitz_index = index
                start_timestamp = dataframe["time"][index]
                second_index = index
                fulfilled = True
                while (dataframe["patient_id"][second_index] == patientid
                       and dataframe["time"][second_index] - start_timestamp <= period_length):
                    if (dataframe["patient_id"][second_index] == patientid
                            and dataframe["time"][second_index] - start_timestamp > series_length):
                        series_length = dataframe["time"][second_index] - start_timestamp
                        series_start = index
                    if dataframe["horovitz"][second_index] >= 300:
                        fulfilled = False
                        break
                    second_index += 1
                    if second_index >= len(dataframe):
                        fulfilled = False
                        break
                if fulfilled:
                    break
                else:
                    horovitz_index = -1
        if horovitz_index != -1:
            return self.add_return_data(return_dataframe, dataframe, horovitz_index)
        elif series_start != -1:
            if (dataframe["ards"][dataframe[dataframe["patient_id"] == patientid].index[0]] == 0
                    or not self.remove_ards_patients_without_onset):
                return self.add_return_data(return_dataframe, dataframe, series_start)
        else:
            if (dataframe["ards"][dataframe[dataframe["patient_id"] == patientid].index[0]] == 0
                    or not self.remove_ards_patients_without_onset):
                horovitz_index = (dataframe[dataframe["patient_id"] == patientid])["horovitz"].idxmin()
                return_dataframe = self.add_return_data(return_dataframe, dataframe, horovitz_index)
                return return_dataframe

    def get_sub_series_indices(self, dataframe, threshold_index):
        """
        Gets the start and end indices for the sub-series around the detected onset.
        
        Args:
            dataframe: DataFrame containing patient data
            threshold_index: Index of the detected onset
            
        Returns:
            Tuple containing the start and end indices of the sub-series
        """
        logger.debug(f"Getting sub-series indices for threshold index: {threshold_index}")
        patientid = dataframe["patient_id"][threshold_index]
        start_index = threshold_index
        end_index = threshold_index
        threshold_timestamp = dataframe["time"][threshold_index]
        for i in reversed(range(0, threshold_index)):
            if (dataframe["time"][i] - threshold_timestamp >= self.return_series_start
                    and dataframe["patient_id"][i] == patientid):
                start_index = i
            else:
                break
        for i in range(threshold_index, len(dataframe)):
            if (dataframe["time"][i] - threshold_timestamp <= self.return_series_end
                    and dataframe["patient_id"][i] == patientid):
                end_index = i
            else:
                break
        logger.debug(f"Sub-series indices determined: start_index={start_index}, end_index={end_index + 1}")
        return start_index, end_index + 1
