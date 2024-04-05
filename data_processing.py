from unit_converter import UnitConverter
from data_imputator import DataImputator
from param_calculation import ParamCalculator
from onset_determiner import OnsetDeterminer
from filter import Filter

class DataProcessor:
    def __init__(self, config, database_name, process):
        self.filter = Filter(config["filtering"])
        self.unit_converter = UnitConverter()
        self.data_imputator = DataImputator(config["imputation"])
        self.param_calculator = ParamCalculator(config["params_to_calculate"])
        self.onset_determiner = OnsetDeterminer(config["ards_onset_detection"], self.data_imputator)
        self.database_name = database_name
        self.process = process

    def process_data(self, dataframe):
        print("Start data preprocessing...")
        if self.process["perform_imputation"] == True:
            print("Impute missing data...")
            dataframe = self.data_imputator.impute_missing_data(dataframe)
            print("Done!")
        if self.process["perform_unit_conversion"] == True:
            print("Convert units...")
            dataframe = self.unit_converter.convert_units(dataframe, self.database_name)
            print("Done!")
        if self.process["calculate_missing_params"] == True:
            print("Calculate missing parameters...")
            dataframe = self.param_calculator.calculate_missing_params(dataframe)
            print("Done!")
        if self.process["perform_ards_onset_detection"] == True:
            print("Detect ARDS onset..")
            dataframe = self.onset_determiner.determine_ards_onset(dataframe)
            print("Done!")
        if self.process["perform_filtering"] == True:
            print("Filter data...")
            dataframe = self.filter.filter_data(dataframe)
            print("Done!")
        print("Data preprocessing finished!")
        return dataframe