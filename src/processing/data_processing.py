
from processing.filter import Filter
from processing.unit_converter import UnitConverter
from processing.data_imputator import DataImputator
from processing.param_calculation import ParamCalculator
from processing.onset_determiner import OnsetDeterminer
from processing.datasets_metadata import TimeseriesMetaData
from processing.ad_algorithms import PhysicalLimitsDetector, SW_ABSAD_Mod_Detector, DeepAntAnomalyDetector
from processing.processing_utils import prepare_multiprocessing, get_processing_meta_data

import pandas as pd
import math
from multiprocessing import Pool




class DataProcessor:
    def __init__(self, config, database_name, process):
        self.filter = Filter(config["filtering"])
        self.patients_per_process = config["patients_per_process"]
        self.max_processes = config["max_processes"]
        self.anomaly_detector = self.init_ad(config["anomaly_detection"], database_name)
        self.unit_converter = UnitConverter()
        self.data_imputator = DataImputator(config["imputation"])
        self.param_calculator = ParamCalculator(config["params_to_calculate"])
        self.onset_determiner = OnsetDeterminer(config["ards_onset_detection"], self.data_imputator)
        self.database_name = database_name
        self.process = process

    def init_ad(self, config, database_name):
        for key, value in config.items():
            if value["active"]:
                value["database"] = database_name
                del value["active"]
                value["max_processes"] = self.max_processes
                if key == "Physical_Outliers":
                    return PhysicalLimitsDetector(**value)
                if key == "SW_ABSAD_Mod":
                    return SW_ABSAD_Mod_Detector(**value)
                if key == "DeepAnt":
                    return DeepAntAnomalyDetector(**value)

    def process_data(self, dataframe: pd.DataFrame, dataset_metadata: TimeseriesMetaData):

        process_pool_data_list, n_jobs = prepare_multiprocessing(dataframe, self.patients_per_process)


        print("Start data preprocessing...")
        if self.process["perform_anomaly_detection"]:
            process_pool_data_list,  n_jobs, dataframe = self.anomaly_detector.run_handler(process_pool_data_list, n_jobs, self.patients_per_process)
            self.anomaly_detector.create_meta_data()


        if self.process["perform_imputation"]:
            print("Impute missing data...")
            with Pool(processes=self.max_processes) as pool:
                process_pool_data_list = pool.starmap(self.data_imputator.impute_missing_data, [(process_pool_data_list[i], i, n_jobs) for i in range(n_jobs)])

            dataframe = pd.concat(process_pool_data_list).reset_index(drop=True)
            self.data_imputator.create_meta_data()
            print("Finished imputing missing data.")


        if self.process["perform_unit_conversion"]:
            if not dataset_metadata or  (dataset_metadata and not dataset_metadata.imputation):
                print("Convert units...")
                columns_to_convert = []
                for column in dataframe.columns:
                    if column in self.unit_converter.conversion_formulas[self.database_name].keys():
                        columns_to_convert.append(column)
                self.unit_converter.columns_to_convert = columns_to_convert

                with Pool(processes=self.max_processes) as pool:
                    process_pool_data_list = pool.starmap(self.unit_converter.convert_units, [(process_pool_data_list[i],   self.database_name, i, n_jobs) for i in range(n_jobs)])

                dataframe = pd.concat(process_pool_data_list).reset_index(drop=True)
                self.unit_converter.create_meta_data(self.database_name)
                print("Converted units!")

            else:
                print("Data is already converted. Skipping...")

        if self.process["calculate_missing_params"]:
            print("Calculate missing parameters...")

            with Pool(processes=self.max_processes) as pool:
                process_pool_data_list = pool.starmap(self.param_calculator.calculate_missing_params,
                                                      [(process_pool_data_list[i], i, n_jobs) for i in range(n_jobs)])
            dataframe = pd.concat(process_pool_data_list).reset_index(drop=True)
            self.param_calculator.create_metadata()
            print("Calculated missing params.")


        if self.process["perform_ards_onset_detection"]:
            if not dataset_metadata or (dataset_metadata and not dataset_metadata.onset_detection):
                print("Detect ARDS onset..")
                with Pool(processes=self.max_processes) as pool:
                    process_pool_data_list = pool.starmap(self.onset_determiner.determine_ards_onset,
                                                          [(process_pool_data_list[i], i, n_jobs) for i in range(n_jobs)])
                dataframe = pd.concat(process_pool_data_list).reset_index(drop=True)
                self.onset_determiner.create_meta_data()
                print("Detected ARDS onset!")
            else:
                print("Data is already converted. Skipping...")

        if self.process["perform_filtering"]:
            print("Filter data...")
            dataframe = self.filter.filter_data(dataframe)
            self.filter.create_meta_data()
            print("Filtered data!")
        print("Data preprocessing completed!")
        return dataframe

    def processing_meta_data(self):
        processing_step_dict = {

            "filtering": self.filter,
            "unit_conversion": self.unit_converter,
            "imputation": self.data_imputator,
            "param_calculation": self.param_calculator,
            "onset_determination": self.onset_determiner,
            "anomaly_detection": self.anomaly_detector
        }
        meta_data_dict = get_processing_meta_data(self.database_name, processing_step_dict)
        return meta_data_dict




