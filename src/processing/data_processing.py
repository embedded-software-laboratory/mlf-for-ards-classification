
from processing.filter import Filter
from processing.unit_converter import UnitConverter
from processing.data_imputator import DataImputator
from processing.param_calculation import ParamCalculator
from processing.onset_determiner import OnsetDeterminer
from processing.datasets_metadata import TimeseriesMetaData

import pandas as pd
import math
from multiprocessing import Pool




class DataProcessor:
    def __init__(self, config, database_name, process):
        self.filter = Filter(config["filtering"])
        self.patients_per_process = config["patients_per_process"]
        self.max_processes = config["max_processes"]
        self.unit_converter = UnitConverter()
        self.data_imputator = DataImputator(config["imputation"])
        self.param_calculator = ParamCalculator(config["params_to_calculate"])
        self.onset_determiner = OnsetDeterminer(config["ards_onset_detection"], self.data_imputator)
        self.database_name = database_name
        self.process = process

    def process_data(self, dataframe: pd.DataFrame, dataset_metadata: TimeseriesMetaData):

        process_pool_data_list, n_jobs = self._prepare_multiprocessing(dataframe)


        print("Start data preprocessing...")
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


    def _prepare_multiprocessing(self, dataframe: pd.DataFrame) -> (list[pd.DataFrame], int):
        patient_ids = list(dataframe["patient_id"].unique())
        num_patients = len(patient_ids)
        n_jobs = math.ceil(num_patients / self.patients_per_process)

        patient_pos_dict = {}
        patient_max_time_df = dataframe.groupby("patient_id")["time"].idxmax().reset_index(drop=True)

        patient_min_time_df = dataframe.groupby("patient_id")["time"].idxmin().reset_index(drop=True)
        for i in range(len(patient_ids)):
            patient_pos_dict[patient_ids[i]] = (int(patient_min_time_df[i]),
                                                int(patient_max_time_df[i])
                                                )
        index = 0
        process_pool_data_list = []
        for i in range(n_jobs):
            first_patient = patient_ids[index + i * self.patients_per_process]
            calculated_last_index = index + (i + 1) * self.patients_per_process - 1
            last_index = calculated_last_index if calculated_last_index < num_patients else num_patients - 1

            last_patient = patient_ids[last_index]
            first_patient_begin_index = patient_pos_dict[first_patient][0]
            last_patient_end_index = patient_pos_dict[last_patient][1]
            split_dataframe = dataframe[first_patient_begin_index:last_patient_end_index]
            process_pool_data_list.append(split_dataframe)

        return process_pool_data_list, n_jobs

    def get_processing_meta_data(self):
        meta_data_dict = {
            "database_name": self.database_name,
            "imputator": self.data_imputator.meta_data,
            "unit_converter": self.unit_converter.meta_data,
            "param_calculator": self.param_calculator.meta_data,
            "onset_determiner": self.onset_determiner.meta_data,
            "filtering": self.filter.meta_data
        }
        return meta_data_dict

