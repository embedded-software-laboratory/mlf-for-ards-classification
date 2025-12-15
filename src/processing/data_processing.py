from processing.data_filter import DataFilter
from processing.unit_converter import UnitConverter
from processing.data_imputator import DataImputator
from processing.param_calculation import ParamCalculator
from processing.onset_determiner import OnsetDeterminer
from processing.datasets_metadata import TimeseriesMetaData
from processing.ad_algorithms import PhysiologicalLimitsDetector, SW_ABSAD_Mod_Detector, DeepAntDetector, BaseAnomalyDetector, ALADDetector
from processing.processing_utils import prepare_multiprocessing, get_processing_meta_data

import pandas as pd
import math
from multiprocessing import Pool
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Main data processing class that orchestrates all preprocessing steps including
    anomaly detection, imputation, unit conversion, parameter calculation, ARDS onset detection,
    and filtering. Uses multiprocessing for efficient handling of large datasets.
    """
    
    def __init__(self, config, database_name, process):
        """
        Initializes the DataProcessor with all necessary preprocessing components.
        
        Args:
            config: Configuration dictionary with preprocessing parameters
            database_name: Name of the database (e.g., 'MIMIC', 'UKA')
            process: Dictionary indicating which processing steps should be executed
        """
        logger.info("Initializing DataProcessor...")
        self.filter = DataFilter(config["filtering"])
        self.patients_per_process = config["patients_per_process"]
        self.max_processes = config["max_processes"]
        logger.info(f"Max processes set to: {self.max_processes}, Patients per process: {self.patients_per_process}")
        
        self.anomaly_detector = self.init_ad(config["anomaly_detection"], database_name)
        self.unit_converter = UnitConverter()
        self.data_imputator = DataImputator(config["imputation"])
        self.param_calculator = ParamCalculator(config["params_to_calculate"])
        self.onset_determiner = OnsetDeterminer(config["ards_onset_detection"], self.data_imputator)
        self.database_name = database_name
        self.process = process
        logger.info(f"DataProcessor initialized with database: {database_name}")

    def init_ad(self, config, database_name):
        """
        Initializes and configures the appropriate anomaly detection algorithm based on configuration.
        
        Args:
            config: Anomaly detection configuration dictionary
            database_name: Name of the database for detector initialization
            
        Returns:
            Initialized anomaly detector object or BaseAnomalyDetector if none active
        """
        logger.info("Initializing anomaly detector...")
        for key, value in config.items():
            if value["active"]:
                logger.info(f"Activating anomaly detection method: {key}")
                value["database"] = database_name
                del value["active"]
                value["max_processes"] = self.max_processes
                
                if key == "Physiological_Outliers":
                    detector = PhysiologicalLimitsDetector(**value)
                    logger.info("Physiological Limits Detector initialized")
                    return detector
                    
                if key == "SW_ABSAD_MOD":
                    detector = SW_ABSAD_Mod_Detector(**value)
                    logger.info("SW_ABSAD_MOD Detector initialized")
                    return detector
                    
                if key == "DeepAnt":
                    detector = DeepAntDetector(**value)
                    logger.info("DeepAnt Detector initialized")
                    return detector
                    
                if key == "ALAD":
                    detector = ALADDetector(**value)
                    logger.info("ALAD Detector initialized")
                    return detector
                    
        logger.warning("No anomaly detection method activated, using BaseAnomalyDetector")
        return BaseAnomalyDetector()

    def process_data(self, dataframe: pd.DataFrame, dataset_metadata: TimeseriesMetaData):
        """
        Main data processing pipeline that applies all configured preprocessing steps
        in sequence using multiprocessing for performance optimization.
        
        Args:
            dataframe: Input DataFrame with raw timeseries data
            dataset_metadata: Metadata object containing information about previous processing
            
        Returns:
            Processed DataFrame with all preprocessing steps applied
        """
        logger.info("=" * 80)
        logger.info("STARTING DATA PREPROCESSING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Input data shape: {dataframe.shape}")
        logger.info(f"Number of unique patients: {dataframe['patient_id'].nunique() if 'patient_id' in dataframe.columns else 'Unknown'}")
        
        # Prepare data for multiprocessing
        process_pool_data_list, n_jobs = prepare_multiprocessing(dataframe, self.patients_per_process, self.max_processes)
        logger.info(f"Data split into {n_jobs} jobs for parallel processing")

        # Step 1: Unit Conversion
        if self.process["perform_unit_conversion"]:
            logger.info("-" * 80)
            logger.info("SUBSTEP 1: Unit Conversion")
            logger.info("-" * 80)
            
            if not dataset_metadata or (dataset_metadata and not dataset_metadata.unit_conversion):
                logger.info(f"Converting units for database: {self.database_name}...")
                columns_to_convert = []
                for column in dataframe.columns:
                    if column in self.unit_converter.conversion_formulas[self.database_name].keys():
                        columns_to_convert.append(column)
                
                logger.info(f"Found {len(columns_to_convert)} columns to convert: {columns_to_convert}")
                self.unit_converter.columns_to_convert = columns_to_convert

                with Pool(processes=self.max_processes) as pool:
                    process_pool_data_list = pool.starmap(
                        self.unit_converter.convert_units,
                        [(process_pool_data_list[i], self.database_name, i+1, n_jobs) for i in range(n_jobs)]
                    )

                dataframe = pd.concat(process_pool_data_list).reset_index(drop=True)
                self.unit_converter.create_meta_data(self.database_name)
                logger.info(f"Unit conversion completed. Data shape: {dataframe.shape}")
            else:
                logger.info("Data already converted in previous run. Skipping unit conversion...")
        else:
            logger.info("SUBSTEP 1: Unit Conversion - SKIPPED (disabled in config)")

        # Step 1: Anomaly Detection
        if self.process["perform_anomaly_detection"]:
            logger.info("-" * 80)
            logger.info("SUBSTEP 2: Anomaly Detection")
            logger.info("-" * 80)
            logger.info(f"Starting anomaly detection with {self.anomaly_detector.__class__.__name__}...")
            process_pool_data_list, n_jobs, dataframe = self.anomaly_detector.execute_handler(
                process_pool_data_list, self.patients_per_process
            )
            self.anomaly_detector.create_meta_data()
            logger.info(f"Anomaly detection completed. Data shape: {dataframe.shape}")
        else:
            logger.info("SUBSTEP 2: Anomaly Detection - SKIPPED (disabled in config)")

        # Step 3: Data Imputation
        if self.process["perform_imputation"]:
            logger.info("-" * 80)
            logger.info("SUBSTEP 3: Data Imputation (Missing Data Handling)")
            logger.info("-" * 80)
            with Pool(processes=self.max_processes) as pool:
                process_pool_data_list = pool.starmap(
                    self.data_imputator.impute_missing_data,
                    [(process_pool_data_list[i], i+1, n_jobs) for i in range(n_jobs)]
                )

            dataframe = pd.concat(process_pool_data_list).reset_index(drop=True)
            self.data_imputator.create_meta_data()
            logger.info(f"Imputation completed successfully. Data shape: {dataframe.shape}")
        else:
            logger.info("SUBSTEP 3: Data Imputation - SKIPPED (disabled in config)")

        # Step 4: Parameter Calculation
        if self.process["calculate_missing_params"]:
            logger.info("-" * 80)
            logger.info("SUBSTEP 4: Missing Parameter Calculation")
            logger.info("-" * 80)

            with Pool(processes=self.max_processes) as pool:
                process_pool_data_list = pool.starmap(
                    self.param_calculator.calculate_missing_params,
                    [(process_pool_data_list[i], i+1, n_jobs) for i in range(n_jobs)]
                )
            
            dataframe = pd.concat(process_pool_data_list).reset_index(drop=True)
            self.param_calculator.create_meta_data()
            logger.info(f"Parameter calculation completed. Data shape: {dataframe.shape}")
        else:
            logger.info("SUBSTEP 4: Missing Parameter Calculation - SKIPPED (disabled in config)")

        # Step 5: ARDS Onset Detection
        if self.process["perform_ards_onset_detection"]:
            logger.info("-" * 80)
            logger.info("SUBSTEP 5: ARDS Onset Detection")
            logger.info("-" * 80)
            
            if not dataset_metadata or (dataset_metadata and not dataset_metadata.onset_detection):
                logger.info("Detecting ARDS onset for each patient...")
                with Pool(processes=self.max_processes) as pool:
                    process_pool_data_list = pool.starmap(
                        self.onset_determiner.determine_ards_onset,
                        [(process_pool_data_list[i], i+1, n_jobs) for i in range(n_jobs)]
                    )
                
                dataframe = pd.concat(process_pool_data_list).reset_index(drop=True)
                self.onset_determiner.create_meta_data()
                logger.info(f"ARDS onset detection completed. Data shape: {dataframe.shape}")
            else:
                logger.info("ARDS onset already detected in previous run. Skipping...")
        else:
            logger.info("SUBSTEP 5: ARDS Onset Detection - SKIPPED (disabled in config)")

        # Step 6: Data Filtering
        if self.process["perform_filtering"]:
            logger.info("-" * 80)
            logger.info("SUBSTEP 6: Data Filtering")
            logger.info("-" * 80)
            dataframe = self.filter.filter_data(dataframe)
            self.filter.create_meta_data()
            logger.info(f"Filtering completed. Data shape after filtering: {dataframe.shape}")
        else:
            logger.info("SUBSTEP 6: Data Filtering - SKIPPED (disabled in config)")

        logger.info("=" * 80)
        logger.info("DATA PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Final data shape: {dataframe.shape}")
        logger.info("=" * 80)
        
        return dataframe

    def processing_meta_data(self):
        """
        Collects and returns metadata from all preprocessing steps performed.
        This metadata tracks which processing steps were executed and their parameters.
        
        Returns:
            Dictionary containing metadata from all preprocessing components
        """
        processing_step_dict = {
            "filtering": self.filter,
            "unit_conversion": self.unit_converter,
            "imputation": self.data_imputator,
            "param_calculation": self.param_calculator,
            "onset_determination": self.onset_determiner,
            "anomaly_detection": self.anomaly_detector
        }
        
        meta_data_dict = get_processing_meta_data(self.database_name, processing_step_dict)
        logger.info(f"Metadata collection completed. Available metadata keys: {list(meta_data_dict.keys())}")
        return meta_data_dict




