import pandas as pd

from processing.datasets_metadata import UnitConversionMetaData


class UnitConverter:

    def __init__(self):
        #self.conversion_formulas_old = {"eICU": {"hemoglobin": "source_value * 0.6206", "creatinine": "source_value * 0.1665", "albumin": "source_value * 151.5152", "crp": "source_value * 9.5238", "bilirubin": "source_value * 17.1037", "etco2": "source_value * 1.7"},
        #                            "MIMIC3": {"hemoglobin": "source_value * 0.6206", "Harnstoff": "source_value * 0.1665", "creatinine": "source_value * 0.1665", "albumin": "source_value * 151.5152", "crp": "source_value * 9.5238", "bilirubin": "source_value * 17.1037"},
        #                           "MIMIC4": {"hemoglobin": "source_value * 0.6206"},
        #                           "UKA": {}}
        self.conversion_formulas = {
            "eICU": {"hemoglobin": (lambda x: x * 0.6206), "creatinine": ( lambda x: x * 0.1665),
                     "albumin": (lambda x: x * 151.5152), "crp": (lambda x: x * 9.5238),
                     "bilirubin": (lambda x: x * 17.1037), "etco2": (lambda x: x * 1.7)},
            "MIMIC3": {"hemoglobin": (lambda x: x * 0.6206), "Harnstoff": (lambda x: x * 0.1665),
                       "creatinine": (lambda x: x * 0.1665), "albumin": (lambda x: x * 151.5152),
                       "crp": (lambda x: x * 9.5238), "bilirubin": (lambda x: x * 17.1037)},
            "MIMIC4": {"hemoglobin": (lambda x: x * 0.6206)},
            "UKA": {}}
        self._columns_to_convert = None
        self.meta_data = None


    def convert_units(self, dataframe: pd.DataFrame, database_name: str, job_number: int, total_job_count: int):
        print("Start unit conversion for job " + str(job_number) + f" of {total_job_count} jobs...")
        #formulas = self.conversion_formulas_old[database_name]
        for column in self._columns_to_convert:
            dataframe[column] = dataframe[column].apply(self.conversion_formulas[database_name][column])
        #for series_name, series in dataframe.items():
        #   if series_name in formulas:
        #        for index in series.index:
        #            formula = formulas[series_name].replace("source_value", str(series[index]))
        #            series[index] = eval(formula)
        print("Finished unit conversion for job " + str(job_number) + f" of {total_job_count} jobs...")
        return dataframe

    def create_meta_data(self, database_name: str):
        meta_data_dict = {}
        for column in self._columns_to_convert:
            meta_data_dict[column] = self.conversion_formulas[database_name][column]
        self.meta_data = UnitConversionMetaData(meta_data_dict)


    @property
    def columns_to_convert(self):
        return self._columns_to_convert

    @columns_to_convert.setter
    def columns_to_convert(self, columns_to_convert):
        self._columns_to_convert = columns_to_convert

