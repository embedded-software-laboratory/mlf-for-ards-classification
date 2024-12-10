import pandas as pd

from processing.datasets_metadata import UnitConversionMetaData


class UnitConverter:

    def __init__(self):
        #self.conversion_formulas_old = {"eICU": {"hemoglobin": "source_value * 0.6206", "creatinine": "source_value * 0.1665", "albumin": "source_value * 151.5152", "crp": "source_value * 9.5238", "bilirubin": "source_value * 17.1037", "etco2": "source_value * 1.7"},
        #                            "MIMIC3": {"hemoglobin": "source_value * 0.6206", "Harnstoff": "source_value * 0.1665", "creatinine": "source_value * 0.1665", "albumin": "source_value * 151.5152", "crp": "source_value * 9.5238", "bilirubin": "source_value * 17.1037"},
        #                           "MIMIC4": {"hemoglobin": "source_value * 0.6206"},
        #                           "UKA": {}}
        self._conversion_formulas = {
            "eICU": {"hemoglobin": self.convert_hemoglobin, "creatinine": self.convert_creatinine,
                     "albumin": self.convert_albumin, "crp":self.convert_crp,
                     "bilirubin": self.convert_bilirubin, "etco2": self.convert_etco2},
            "MIMIC3": {"hemoglobin": self.convert_hemoglobin, "Harnstoff": self.convert_harnstoff,
                       "creatinine": self.convert_creatinine, "albumin": self.convert_albumin, "crp":self.convert_crp,
                       "bilirubin": self.convert_bilirubin},
            "MIMIC4": {"hemoglobin": self.convert_hemoglobin},
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
        self.meta_data = UnitConversionMetaData(conversions=meta_data_dict)

    @property
    def conversion_formulas(self):
        return self._conversion_formulas


    @staticmethod
    def convert_hemoglobin(value):
        return value * 0.6206

    @staticmethod
    def convert_creatinine(value):
        return value * 0.1665

    @staticmethod
    def convert_harnstoff(value):
        return value * 0.1665

    @staticmethod
    def convert_albumin(value):
        return value * 151.5152

    @staticmethod
    def convert_crp(value):
        return value * 9.5238

    @staticmethod
    def convert_bilirubin(value):
        return value * 17.1037

    @staticmethod
    def convert_etco2(value):
        return value * 1.7


    @property
    def columns_to_convert(self):
        return self._columns_to_convert

    @columns_to_convert.setter
    def columns_to_convert(self, columns_to_convert):
        self._columns_to_convert = columns_to_convert

