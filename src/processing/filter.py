import sys

class Filter():

    def __init__(self, config) -> None:
        self.filter = []
        self.available_filter = ["A", "B", "C"]
        self.set_filter(config["filter"])

    def filter_data(self, dataframe):
        for filter in self.filter:
            if filter == "A": 
                dataframe = self.filter_a(dataframe)
            if filter == "B": 
                dataframe = self.filter_b(dataframe)
            if filter == "C": 
                dataframe = self.filter_c(dataframe)
        return dataframe

    def filter_a(self, dataframe):
        for patient_id in dataframe["patient_id"].unique():
            patient_subframe = dataframe[dataframe["patient_id"] == patient_id]
            if not 1 in patient_subframe["ards"].values:
                if (patient_subframe["horovitz"] < 200).any():
                    dataframe = dataframe.drop(patient_subframe.index)
        return dataframe
    
    def filter_b(self, dataframe):
        if "hypervolemia" not in dataframe.columns or "pulmonary-edema" not in dataframe.columns or "hypervolemia" not in dataframe.columns:
            print("Skipping filter b since not all necessary columns are present")
            return dataframe
        for patient_id in dataframe["patient_id"].unique():
            patient_subframe = dataframe[dataframe["patient_id"] == patient_id]
            if not 1 in patient_subframe["ards"].values:
                if not (patient_subframe[dataframe["horovitz"] < 200]).empty:
                    if not 1 in patient_subframe["hypervolemia"].values and not 1 in patient_subframe["pulmonary-edema"].values and not 1 in patient_subframe["heart-failure"].values:
                        dataframe = dataframe.drop(patient_subframe.index)
        return dataframe
    
    def filter_c(self, dataframe):
        for patient_id in dataframe["patient_id"].unique():
            patient_subframe = dataframe[dataframe["patient_id"] == patient_id]
            if 1 in patient_subframe["ards"].values:
                if (patient_subframe[dataframe["horovitz"] < 300][dataframe["peep"] >= 5]).empty:
                    dataframe = dataframe.drop(patient_subframe.index)
        return dataframe


    def set_filter(self, filter_list):
        for filter in filter_list:
            if filter not in self.available_filter:
                raise RuntimeError("Filter " + filter + " not available. Avaliable are " + str(self.available_filter))
        self.filter = filter_list