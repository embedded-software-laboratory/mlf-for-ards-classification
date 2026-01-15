import logging
import numpy as np
import pandas as pd
from processing.datasets_metadata import ParamCalculationMetaData

logger = logging.getLogger(__name__)

class ParamCalculator:
    def __init__(self, params_to_calculate):
        """
        Initializes the ParamCalculator with parameters to calculate.
        
        Args:
            params_to_calculate: List of parameters to calculate
        """
        logger.info("Initializing ParamCalculator...")
        self.params_to_calculate = ["horovitz"]
        self.possible_params = ["delta-p", "tidal-vol-per-kg", "liquid-balance", "lymphocytes (absolute)", "horovitz", "i-e", "lymphocytes (percentage)", "age", "admission period"]
        for param in params_to_calculate:
            self.add_param_to_calculate(param)
        self.calculated_params = set()
        self.skipped_params = set()
        self.meta_data = None
        logger.info("ParamCalculator initialized successfully.")

    def calculate_missing_params(self, dataframe, job_number: int, total_job_count: int) -> pd.DataFrame:
        """
        Calculates missing parameters in the provided DataFrame.
        Parameters that cannot be calculated due to missing dependencies are skipped gracefully.
        
        Args:
            dataframe: DataFrame containing patient data
            job_number: Current job number for logging
            total_job_count: Total number of jobs for logging
            
        Returns:
            Updated DataFrame with calculated parameters
        """
        logger.info(f"Start parameter calculation for job {job_number} of {total_job_count} jobs...")

        for param in self.params_to_calculate:
            if param in dataframe.columns:
                logger.debug(f"Parameter '{param}' already exists in dataset, skipping calculation")
            else:
                try:
                    if param == "delta-p":
                        dataframe = self.calculate_delta_p(dataframe)
                    elif param == "tidal-vol-per-kg":
                        dataframe = self.calculate_individual_tv(dataframe)
                    elif param == "liquid-balance":
                        dataframe = self.calculate_fluid_balance(dataframe)
                    elif param == "lymphocytes (absolute)":
                        dataframe = self.calculate_absolute_lymphocytes(dataframe)
                    elif param == "horovitz":
                        dataframe = self.calculate_horovitz(dataframe)
                    elif param == "i-e":
                        dataframe = self.calculate_ie_ratio(dataframe)
                    elif param == "lymphocytes (percentage)":
                        dataframe = self.calculate_lymphocyte_percentage(dataframe)
                    
                    self.calculated_params.add(param)
                    logger.debug(f"Successfully calculated parameter: {param}")
                    
                except RuntimeError as e:
                    logger.warning(f"Skipping calculation of '{param}': {str(e)}")
                    self.skipped_params.add(param)
                except Exception as e:
                    logger.warning(f"Unexpected error calculating '{param}': {str(e)}. Skipping this parameter.")
                    self.skipped_params.add(param)

        logger.info(f"Finished parameter calculation for job {job_number} of {total_job_count} jobs. "
                   f"Calculated: {len(self.calculated_params)}, Skipped: {len(self.skipped_params)}")
        return dataframe

    def create_meta_data(self):
        """
        Creates metadata for the calculated parameters.
        """
        if len(list(self.calculated_params)) > 0:
            logger.info(f"Creating metadata for calculated parameters: {list(self.calculated_params)}")
            if len(self.skipped_params) > 0:
                logger.info(f"Skipped parameters due to missing dependencies: {list(self.skipped_params)}")
            self.meta_data = ParamCalculationMetaData(calculated_parameters=list(self.calculated_params))
            logger.info("Metadata created successfully.")
        else:
            logger.warning("No parameters were calculated. Metadata will not be created.")

    @staticmethod
    def calculate_delta_p(dataframe):
        """
        Calculates the delta P value for the DataFrame.
        
        Args:
            dataframe: DataFrame containing patient data
            
        Returns:
            Updated DataFrame with delta P calculated
            
        Raises:
            RuntimeError: If required parameters are missing
        """
        if "p-ei" not in dataframe.columns:
            raise RuntimeError("P EI is required to calculate delta P, but this parameter is missing in the given dataset.")
        if "peep" not in dataframe.columns:
            raise RuntimeError("PEEP is required to calculate delta P, but this parameter is missing in the given dataset.")
        
        logger.debug("Calculating delta P...")
        delta_p_values = [dataframe["p-ei"][i] - dataframe["peep"][i] for i in dataframe["p-ei"].index]
        dataframe["delta-p"] = delta_p_values
        logger.debug("Delta P calculation completed.")
        return dataframe

    @staticmethod
    def calculate_individual_tv(dataframe):
        """
        Calculates individual tidal volume per kg for the DataFrame.
        
        Args:
            dataframe: DataFrame containing patient data
            
        Returns:
            Updated DataFrame with individual tidal volume calculated
            
        Raises:
            RuntimeError: If required parameters are missing
        """
        if "height" not in dataframe.columns:
            raise RuntimeError("Body height is required to calculate individual tidal volume, but this parameter is missing in the given dataset.")
        if "tidal-volume" not in dataframe.columns:
            raise RuntimeError("Tidal volume is required to calculate individual tidal volume, but this parameter is missing in the given dataset.")
        if "gender" not in dataframe.columns:
            raise RuntimeError("Gender is required to calculate individual tidal volume, but this parameter is missing in the given dataset.")

        logger.debug("Calculating individual tidal volume per kg...")
        # Calculate ideal body weight based on gender
        gender_value = dataframe["gender"].iloc[0]
        height_value = dataframe["height"].iloc[0]
        
        if gender_value == 1:  # Male (mapped to 1)
            ideal_body_weight = 50 + (height_value - 152.4)
        else:  # Female (mapped to 0)
            ideal_body_weight = 45.5 + (height_value - 152.4)
        
        individual_tidal_volumes = [dataframe["tidal-volume"][i] / ideal_body_weight for i in dataframe["tidal-volume"].index]
        dataframe["tidal-vol-per-kg"] = individual_tidal_volumes
        logger.debug("Individual tidal volume calculation completed.")
        return dataframe

    @staticmethod
    def calculate_fluid_balance(dataframe):
        """
        Calculates the fluid balance for the DataFrame.
        
        Args:
            dataframe: DataFrame containing patient data
            
        Returns:
            Updated DataFrame with fluid balance calculated
            
        Raises:
            RuntimeError: If required parameters are missing
        """
        if "liquid-input" not in dataframe.columns:
            raise RuntimeError("Fluid input is required to calculate fluid balance, but this parameter is missing in the given dataset.")
        if "liquid-output" not in dataframe.columns:
            raise RuntimeError("Fluid output is required to calculate fluid balance, but this parameter is missing in the given dataset.")
        
        logger.debug("Calculating fluid balance...")
        inputs, outputs, fluid_balances = [], [], []
        time_start = dataframe["time"].iloc[0]
        index_start = 0
        current_patient = dataframe["patient_id"].iloc[0]

        for i in dataframe["time"].index:
            if dataframe["time"][i] - time_start > 86400 or dataframe["patient_id"][i] != current_patient:
                balance = sum(inputs) - sum(outputs)
                fluid_balances.extend([balance] * (i - index_start))
                index_start = i
                time_start = dataframe["time"][i]
                current_patient = dataframe["patient_id"][i]
                inputs, outputs = [], []

            inputs.append(dataframe["liquid-input"][i])
            outputs.append(dataframe["liquid-output"][i])

        balance = sum(inputs) - sum(outputs)
        fluid_balances.extend([balance] * (len(dataframe["time"]) - index_start))
        dataframe["liquid-balance"] = fluid_balances
        logger.debug("Fluid balance calculation completed.")
        return dataframe

    @staticmethod
    def calculate_absolute_lymphocytes(dataframe):
        """
        Calculates the absolute number of lymphocytes for the DataFrame.
        
        Args:
            dataframe: DataFrame containing patient data
            
        Returns:
            Updated DataFrame with absolute lymphocytes calculated
            
        Raises:
            RuntimeError: If required parameters are missing
        """
        if "lymphocytes (percentage)" not in dataframe.columns:
            raise RuntimeError("Lymphocyte percentage is required to calculate absolute number of lymphocytes, but this parameter is missing in the given dataset.")
        if "leucocytes" not in dataframe.columns:
            raise RuntimeError("Number of leucocytes is required to calculate absolute number of lymphocytes, but this parameter is missing in the given dataset.")
        
        logger.debug("Calculating absolute lymphocytes...")
        lymphocyte_values = [dataframe["lymphocytes (percentage)"][i] * dataframe["leucocytes"][i] for i in dataframe["lymphocytes (percentage)"].index]
        dataframe["lymphocytes (absolute)"] = lymphocyte_values
        logger.debug("Absolute lymphocytes calculation completed.")
        return dataframe

    @staticmethod
    def calculate_horovitz(dataframe):
        """
        Calculates the Horovitz value for the DataFrame.
        
        Args:
            dataframe: DataFrame containing patient data
            
        Returns:
            Updated DataFrame with Horovitz calculated
            
        Raises:
            RuntimeError: If required parameters are missing
        """
        if "pao2" not in dataframe.columns:
            raise RuntimeError("PaO2 is required to calculate horovitz, but this parameter is missing in the given dataset.")
        if "fio2" not in dataframe.columns:
            raise RuntimeError("FiO2 is required to calculate horovitz, but this parameter is missing in the given dataset.")
        
        logger.debug("Calculating horovitz...")
        horovitz_values = []
        debug_count = 0
        negative_count = 0
        for i in dataframe["pao2"].index:
            last_FiO2 = None
            j = i
            while last_FiO2 is None and j >= 0:
                if not np.isnan(dataframe["fio2"][j]):
                    last_FiO2 = dataframe["fio2"][j]
                j -= 1
            if last_FiO2 is not None and last_FiO2 != 0:
                horovitz = dataframe["pao2"][i] / last_FiO2
                horovitz_values.append(horovitz)
                if debug_count < 10:  # Log first 10 calculations
                    logger.debug(f"Row {i}: PaO2={dataframe['pao2'][i]}, FiO2={last_FiO2}, Horovitz={horovitz}")
                    debug_count += 1
                if horovitz < 0:
                    negative_count += 1
                    if negative_count <= 5:  # Log first 5 negative cases
                        logger.warning(f"Negative Horovitz at row {i}: PaO2={dataframe['pao2'][i]}, FiO2={last_FiO2}, Horovitz={horovitz}")
            else:
                horovitz_values.append(np.nan)
        if negative_count > 0:
            logger.warning(f"Total negative Horovitz values: {negative_count}")
        dataframe["horovitz"] = horovitz_values
        logger.debug("Horovitz calculation completed.")
        
        # Log summary statistics for Horovitz values
        logger.debug("Horovitz calculation summary:")
        logger.debug(f"Total rows: {len(dataframe)}")
        logger.debug(f"Valid Horovitz values: {dataframe['horovitz'].notna().sum()}")
        logger.debug(f"NaN Horovitz values: {dataframe['horovitz'].isna().sum()}")
        logger.debug(f"Horovitz describe:\n{dataframe['horovitz'].describe()}")
        
        return dataframe

    @staticmethod
    def calculate_ie_ratio(dataframe):
        """
        Calculates the I:E ratio for the DataFrame.
        
        Args:
            dataframe: DataFrame containing patient data
            
        Returns:
            Updated DataFrame with I:E ratio calculated
            
        Raises:
            RuntimeError: If required parameters are missing
        """
        if "inspiry-time" not in dataframe.columns:
            raise RuntimeError("Inspiratory time is required to calculate I:E ratio, but this parameter is missing in the given dataset.")
        if "expiry-time" not in dataframe.columns:
            raise RuntimeError("Expiratory time is required to calculate I:E ratio, but this parameter is missing in the given dataset.")
        
        logger.debug("Calculating I:E ratio...")
        ie_values = [dataframe["inspiry-time"][i] / dataframe["expiry-time"][i] for i in dataframe["inspiry-time"].index]
        dataframe["i-e"] = ie_values
        logger.debug("I:E ratio calculation completed.")
        return dataframe

    @staticmethod
    def calculate_lymphocyte_percentage(dataframe):
        """
        Calculates the lymphocyte percentage for the DataFrame.
        
        Args:
            dataframe: DataFrame containing patient data
            
        Returns:
            Updated DataFrame with lymphocyte percentage calculated
            
        Raises:
            RuntimeError: If required parameters are missing
        """
        if "lymphocytes (absolute)" not in dataframe.columns:
            raise RuntimeError("Absolute number of lymphocytes is required to calculate lymphocyte percentage, but this parameter is missing in the given dataset.")
        if "leucocytes" not in dataframe.columns:
            raise RuntimeError("Number of leucocytes is required to calculate lymphocyte percentage, but this parameter is missing in the given dataset.")
        
        logger.debug("Calculating lymphocyte percentage...")
        lymphocyte_values = [dataframe["lymphocytes (absolute)"][i] / dataframe["leucocytes"][i] for i in dataframe["lymphocytes (absolute)"].index]
        dataframe["lymphocytes (percentage)"] = lymphocyte_values
        logger.debug("Lymphocyte percentage calculation completed.")
        return dataframe

    def add_param_to_calculate(self, param):
        """
        Adds a parameter to the list of parameters to calculate.
        
        Args:
            param: Parameter to add
            
        Raises:
            RuntimeError: If the parameter is not supported
        """
        logger.debug(f"Adding parameter to calculate: {param}")
        if param in self.possible_params:
            if param not in self.params_to_calculate:
                self.params_to_calculate.append(param)
                logger.debug(f"Parameter '{param}' added successfully.")
        else:
            error_msg = f"Parameter '{param}' is currently not supported/not known and cannot be calculated."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def remove_param_to_calculate(self, param):
        """
        Removes a parameter from the list of parameters to calculate.
        
        Args:
            param: Parameter to remove
        """
        logger.debug(f"Removing parameter from calculation: {param}")
        if param in self.params_to_calculate:
            self.params_to_calculate.remove(param)
            logger.debug(f"Parameter '{param}' removed successfully.")

    def set_params_to_calculate(self, param_list):
        """
        Sets the parameters to calculate based on the provided list.
        
        Args:
            param_list: List of parameters to set
            
        Raises:
            RuntimeError: If any parameter is not supported
        """
        logger.info("Setting parameters to calculate...")
        for param in param_list:
            if param not in self.possible_params:
                error_msg = f"Parameter '{param}' is currently not supported/not known and cannot be calculated."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        self.params_to_calculate = param_list
        logger.info(f"Parameters set successfully: {param_list}")
