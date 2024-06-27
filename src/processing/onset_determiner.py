import pandas as pd
import numpy as np


class OnsetDeterminer:  #class to determine the ARDS onset in a given dataset
    def __init__(self, config, data_imputator) -> None:
        self.detection_rule = "first_horovitz"
        # The available methods to determine the ards onset in the data (the peep must be at least 5 for all horovitz values)
        # 4h / 12h / 24h: The first timestamp at which the horovitz is under 300 for the next 4 / 12 / 24 hours
        # 4h_50: The first timestamp at which the horovitz is under 300 for 50% of the values for the next 4 hours
        # lowest_horovitz: the lowest horovitz in the given data
        # first_horovitz: the first horovitz which is under 300
        self.available_detection_rules = ["4h", "4h_50", "12h", "24h", "lowest_horovitz", "first_horovitz"]
        self.return_rule = "datapoint"
        self.available_return_rules = ["datapoint", "data_series_as_series", "data_series_as_point"]
        self.return_series_start = None  #in seconds, relative to the determined ards onset. -500 means, 500 seconds before the ards onset
        self.return_series_end = None  #in seconds, relative to the determined ards onset. 600 means, 600 seconds after the ards onset
        self.set_detection_rule(config["detection_rule"])
        self.set_return_rule(config["return_rule"], config["series_start_point"], config["series_end_point"])
        self.remove_ards_patients_without_onset = config["remove_ards_patients_without_onset"]
        self.impute_missing_rows = config["impute_missing_rows"]
        self.data_imputator = data_imputator
        self.update_ards_values = config["update_ards_values"]

    def set_detection_rule(self, rule):
        if rule not in self.available_detection_rules:
            raise RuntimeError(
                "Detection rule " + rule + " is currently not implemented for ARDS onset detection. Available rules are " + str(
                    self.available_detection_rules))
        self.detection_rule = rule

    def set_return_rule(self, rule, series_start=None, series_end=None):
        if rule not in self.available_return_rules:
            raise RuntimeError(
                "Return rule " + rule + " is currently not implemented for ARDS onset detection. Available rules are " + str(
                    self.available_return_rules))
        if (rule == "data_series_as_series" or rule == "data_series_as_point") and (
                series_start == None or series_end == None):
            raise RuntimeError("series length not defined but needed for returning a data series")
        if series_end < series_start:
            raise ValueError("The end of the series must be after the beginning of the series")
        self.return_rule = rule
        self.return_series_start = series_start
        self.return_series_end = series_end

    def determine_ards_onset(self, dataframe):
        return_dataframe = pd.DataFrame()
        for patientid in set(dataframe["patient_id"].to_list()):
            if self.detection_rule == "first_horovitz":
                horovitz_index = -1
                for index in dataframe[dataframe["patient_id"] == patientid].index:
                    if dataframe["horovitz"][index] < 300 and dataframe["peep"][index] >= 5:
                        horovitz_index = index
                        break
                if horovitz_index != -1:
                    return_dataframe = self.add_return_data(return_dataframe, dataframe, horovitz_index)
                else:
                    if (dataframe["ards"][dataframe[dataframe["patient_id"] == patientid].index[0]] == 0
                            or self.remove_ards_patients_without_onset == False):
                        horovitz_index = (dataframe[dataframe["patient_id"] == patientid])["horovitz"].idxmin()
                        return_dataframe = self.add_return_data(return_dataframe, dataframe, horovitz_index)

            if self.detection_rule == "lowest_horovitz":
                horovitz_index = (dataframe[dataframe["patient_id"] == patientid])["horovitz"].idxmin()
                return_dataframe = self.add_return_data(return_dataframe, dataframe, horovitz_index)

            if self.detection_rule == "4h":
                return_dataframe = self.get_low_horovitz_period_start(dataframe, 14400, patientid, return_dataframe)

            if self.detection_rule == "12h":
                return_dataframe = self.get_low_horovitz_period_start(dataframe, 43200, patientid, return_dataframe)

            if self.detection_rule == "24h":
                return_dataframe = self.get_low_horovitz_period_start(dataframe, 86400, patientid, return_dataframe)

            if self.detection_rule == "4h_50":
                horovitz_index = -1
                start_timestamp = 0
                start_index_of_closest_series = -1  # if we do not find a 4h long series which at least 50% horovitz values under 300, use the series which has the most horovitz values under 300
                horovitz_percentage_of_closest_series = 0
                for index in dataframe[dataframe["patient_id"] == patientid].index:
                    if dataframe["horovitz"][index] < 300 and dataframe["peep"][index] >= 5:
                        horovitz_index = index
                        start_timestamp = dataframe["time"][index]
                        second_index = index
                        number_of_low_horovitz = 1
                        total_number_of_rows = 1
                        fulfilled = True  #set to false if we find a horovitz above 300 in the next 12 hours - this breaks our ards onset condition
                        while (dataframe["patient_id"][second_index] == patientid
                               and dataframe["time"][second_index] - start_timestamp <= 14400):
                            total_number_of_rows += 1
                            if dataframe["horovitz"][second_index] < 300:
                                number_of_low_horovitz += 1
                            second_index += 1
                            if second_index >= len(dataframe):
                                fulfilled = False
                                break
                        if number_of_low_horovitz / total_number_of_rows <= 0.5:
                            fulfilled = False
                            if number_of_low_horovitz / total_number_of_rows > horovitz_percentage_of_closest_series:
                                horovitz_percentage_of_closest_series = number_of_low_horovitz / total_number_of_rows > horovitz_percentage_of_closest_series
                                start_index_of_closest_series = horovitz_index
                        if fulfilled == True:
                            break
                        else:
                            horovitz_index = -1
                if horovitz_index != -1:
                    return_dataframe = self.add_return_data(return_dataframe, dataframe, horovitz_index)
                elif start_index_of_closest_series != -1:
                    if dataframe["ards"][dataframe[dataframe["patient_id"] == patientid].index[
                        0]] == 0 or self.remove_ards_patients_without_onset == False:
                        return self.add_return_data(return_dataframe, dataframe, start_index_of_closest_series)
                else:
                    if (dataframe["ards"][dataframe[dataframe["patient_id"] == patientid].index[0]] == 0
                            or self.remove_ards_patients_without_onset is False):
                        horovitz_index = (dataframe[dataframe["patient_id"] == patientid])["horovitz"].idxmin()
                        return_dataframe = self.add_return_data(return_dataframe, dataframe, horovitz_index)

        return return_dataframe

    def add_return_data(self, return_dataframe, dataframe, index):
        if self.return_rule == "datapoint":
            return_dataframe = pd.concat([return_dataframe,
                                          dataframe.iloc[[index]]], axis=0)
            return return_dataframe
        if self.return_rule == "data_series_as_series":
            start_index, end_index = self.get_sub_series_indices(dataframe, index)
            new_dataframe = dataframe.iloc[start_index:end_index]
            if self.impute_missing_rows:
                threshold_timestamp = dataframe["time"][index]
                new_dataframe = self.data_imputator.impute_rows(new_dataframe,
                                                                threshold_timestamp + self.return_series_start,
                                                                threshold_timestamp + self.return_series_end)
            if self.update_ards_values == True and dataframe["ards"][index] == 1:
                for i in new_dataframe.index:
                    if i < index:
                        new_dataframe["ards"][i] = 0
                    else:
                        new_dataframe["ards"][i] = 1
            return_dataframe = pd.concat([return_dataframe, new_dataframe], axis=0, ignore_index=True)
            return return_dataframe
        if self.return_rule == "data_series_as_point":
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
            new_dataframe = new_dataframe.drop(["patient_id", "ards", "time"],
                                               axis=1)  #drop these columns because it makes no sense to have patientid0, patientid1... columns.
            counter = 0
            for i in range(start_index, end_index):
                column_names = [columnname + str(counter) for columnname in new_dataframe.columns]
                values = new_dataframe.loc[i].to_list()
                temp_dataframe2 = pd.DataFrame([values], columns=column_names)
                temp_dataframe = pd.concat([temp_dataframe, temp_dataframe2], axis=1)
                counter += 1
            return_dataframe = pd.concat([return_dataframe, temp_dataframe], axis=0)
            return return_dataframe

    def get_low_horovitz_period_start(self, dataframe, period_length, patientid, return_dataframe):
        horovitz_index = -1
        start_timestamp = 0
        series_length = 0
        series_start = -1
        for index in dataframe[dataframe["patient_id"] == patientid].index:
            if dataframe["horovitz"][index] < 300 and dataframe["peep"][index] >= 5:
                horovitz_index = index
                start_timestamp = dataframe["time"][index]
                second_index = index
                fulfilled = True  # set to false if we find a horovitz above 300 in the next 12 hours - this breaks
                # our ards onset condition
                while (dataframe["patient_id"][second_index] == patientid
                       and dataframe["time"][second_index] - start_timestamp <= period_length):
                    if (dataframe["patient_id"][second_index] == patientid
                            and dataframe["time"][second_index] - start_timestamp > series_length):
                        series_length = dataframe["patient_id"][second_index] == patientid and dataframe["time"][
                            second_index] - start_timestamp
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
                    or self.remove_ards_patients_without_onset == False):
                return self.add_return_data(return_dataframe, dataframe, series_start)
        else:
            if (dataframe["ards"][dataframe[dataframe["patient_id"] == patientid].index[0]] == 0
                    or self.remove_ards_patients_without_onset == False):
                horovitz_index = (dataframe[dataframe["patient_id"] == patientid])["horovitz"].idxmin()
                return_dataframe = self.add_return_data(return_dataframe, dataframe, horovitz_index)
                return return_dataframe

    def get_sub_series_indices(self, dataframe, threshold_index):
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
        return start_index, end_index + 1
