import pandas as pd


class UnitConverter:

    def __init__(self):
        self.conversion_formulas = {"eICU": {"hemoglobin": "source_value * 0.6206", "creatinine": "source_value * 0.1665", "albumin": "source_value * 151.5152", "crp": "source_value * 9.5238", "bilirubin": "source_value * 17.1037", "etco2": "source_value * 1.7"},
                                    "MIMIC3": {"hemoglobin": "source_value * 0.6206", "Harnstoff": "source_value * 0.1665", "creatinine": "source_value * 0.1665", "albumin": "source_value * 151.5152", "crp": "source_value * 9.5238", "bilirubin": "source_value * 17.1037"},
                                    "MIMIC4": {"hemoglobin": "source_value * 0.6206"},
                                    "UKA": {}}
        self.columns_to_convert = None


    def convert_units(self, dataframe: pd.DataFrame, columns_to_convert: list[str], database_name: str, job_number: int, total_job_count: int):
        print("Start unit conversion for job " + str(job_number) + f" of {total_job_count} jobs...")
        formulas = self.conversion_formulas[database_name]
        for column in columns_to_convert:

        for series_name, series in dataframe.items():
            if series_name in formulas:
                for index in series.index:
                    formula = formulas[series_name].replace("source_value", str(series[index]))
                    series[index] = eval(formula)
        print("Finished unit conversion for job " + str(job_number) + f" of {total_job_count} jobs...")
        return dataframe


