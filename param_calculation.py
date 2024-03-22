import numpy as np

class ParamCalculator:

    def __init__(self, params_to_calculate):
        self.params_to_calculate = ["horovitz"]
        self.possible_params = ["delta-p", "tidal-vol-per-kg", "liquid-balance", "lymphocytes (absolute)", "horovitz", "i-e", "lymphocytes (percentage)", "age", "admission period"]
        self.set_params_to_calculate(params_to_calculate)

    def calculate_missing_params(self, dataframe):
        for param in self.params_to_calculate:
            if param == "delta-p": 
                dataframe = self.calculate_delta_p(dataframe)
            if param == "tidal-vol-per-kg":
                dataframe = self.calculate_individual_tv(dataframe)
            if param == "liquid-balance":
                dataframe = self.calculate_fluid_balance(dataframe)
            if param == "lymphocytes (absolute)":
                dataframe = self.calculate_absolute_lymphocytes(dataframe)
            if param == "horovitz":
                dataframe = self.calculate_horovitz(dataframe)
            if param == "i-e":
                dataframe = self.calculate_ie_ratio(dataframe)
            if param == "lymphocytes (prozentual)":
                dataframe = self.calculate_lymphocyte_percentage(dataframe)
        return dataframe


    def calculate_delta_p(self, dataframe):
        if "delta-p" in dataframe.columns:
            print("Skipping calculation of delta P since it already exists in the given dataset")
            return dataframe
        if "p-ei" not in dataframe.columns:
            raise RuntimeError("P EI is required to calculate delta P, but this parameter is missing in the given dataset.")
        if "PEEP" not in dataframe.columns:
            raise RuntimeError("peep is required to calculate delta P, but this parameter is missing in the given dataset.")
        delta_p_values = []
        for i in dataframe["p-ei"].index:
            delta_p_values.append(dataframe["p-ei"][i] - dataframe["peep"][i])
        dataframe["delta-p"] = delta_p_values
        return dataframe
    
    def calculate_individual_tv(self, dataframe):
        if "tidal-vol-per-kg" in dataframe.columns:
            print("Skipping calculation of individual tidal volume since it already exists in the given dataset")
            return dataframe
        if "height" not in dataframe.columns:
            raise RuntimeError("Body height is required to calculate individual tidal volume, but this parameter is missing in the given dataset.")
        if "weight" not in dataframe.columns:
            raise RuntimeError("Weight is required to calculate individual tidal volume, but this parameter is missing in the given dataset.")
        if "tidal-volume" not in dataframe.columns:
            raise RuntimeError("Tidal volume is required to calculate individual tidal volume, but this parameter is missing in the given dataset.")
        ideal_body_weight = 0
        if "gender" in dataframe.columns:
            if dataframe["gender"][0] == "W":
                ideal_body_weight = 45.5+(dataframe["height"][0] - 152.4)
            else:
                ideal_body_weight = 50+(dataframe["height"][0] - 152.4)
        else:
            ideal_body_weight = 50+(dataframe["height"][0] - 152.4)
        individual_tidal_volumes = []
        for i in dataframe["tidal-volume"].index:
            individual_tidal_volumes.append(dataframe["tidal-volume"][i] / ideal_body_weight)
        dataframe["tidal-vol-per-kg"] = individual_tidal_volumes
        return dataframe
    
    def calculate_fluid_balance(self, dataframe):
        if "liquid-balance" in dataframe.columns:
            print("Skipping calculation of fluid balance since it already exists in the given dataset")
            return dataframe
        if "liquid-input" not in dataframe.columns:
            raise RuntimeError("Fluid input is required to calculate fluid balance, but this parameter is missing in the given dataset.")
        if "liquid-output" not in dataframe.columns:
            raise RuntimeError("Fluid output is required to calculate fluid balance, but this parameter is missing in the given dataset.")
        inputs = []
        outputs = []
        time_start = dataframe["time"][0]
        index_start = 0
        fluid_balances = []
        current_patient = dataframe["patient_id"][0]
        for i in dataframe["time"].index:
            if dataframe["time"][i] - time_start > 86400 or dataframe["patient_id"][i] != current_patient: #calculate the balance for 24h
                balance = sum(inputs) - sum(outputs)
                for j in range(index_start, i):
                    fluid_balances.append(balance)
                index_start = i
                time_start = dataframe["time"][i]
                current_patient = dataframe["patient_id"][i]
                inputs = []
                outputs = []

            inputs.append(dataframe["liquid-input"][i])
            outputs.append(dataframe["liquid-output"][i])
        balance = sum(inputs) - sum(outputs)
        for j in range(index_start, len(dataframe["time"])):
            fluid_balances.append(balance)
        dataframe["liquid-balance"] = fluid_balances
        return dataframe
    
    def calculate_absolute_lymphocytes(self, dataframe):
        if "lymphocytes_abs" in dataframe.columns:
            print("Skipping calculation of lymphocytes since it already exists in the given dataset")
            return dataframe
        if "lymphocytes (percentage)" not in dataframe.columns:
            raise RuntimeError("Lymphocyte percentage is required to calculate absolute number of lymphocytes, but this parameter is missing in the given dataset.")
        if "leucocytes" not in dataframe.columns:
            raise RuntimeError("Number of leucocytes is required to calculate absolute number of lymphocytes, but this parameter is missing in the given dataset.")
        lymphocyte_values = []
        for i in dataframe["lymphocytes (percentage)"].index:
            lymphocyte_values.append(dataframe["lymphocytes (percentage)"][i] * dataframe["leucocytes"][i])
        dataframe["lymphocytes_abs"] = lymphocyte_values
        return dataframe
    
    def calculate_horovitz(self, dataframe):
        if "horovitz" in dataframe.columns:
            print("Skipping calculation of horovitz since it already exists in the given dataset")
            return dataframe
        if "pao2" not in dataframe.columns:
            raise RuntimeError("PaO2 is required to calculate horovitz, but this parameter is missing in the given dataset.")
        if "fio2" not in dataframe.columns:
            raise RuntimeError("FiO2 is required to calculate horovitz, but this parameter is missing in the given dataset.")
        horovitz_values = []
        for i in dataframe["pao2"].index:
            last_FiO2 = None
            j = i
            while last_FiO2 == None:
                if not np.isnan(dataframe["fio2"][j]):
                    last_FiO2 = dataframe["fio2"][j]
                j-=1
            horovitz_values.append(dataframe["pao2"][i] / last_FiO2)
            last_FiO2 = None
        dataframe["horovitz"] = horovitz_values
        return dataframe
    
    def calculate_ie_ratio(self, dataframe):
        if "i-e" in dataframe.columns:
            print("Skipping calculation of I:E ratio since it already exists in the given dataset")
            return dataframe
        if "inspiry-time" not in dataframe.columns:
            raise RuntimeError("Inspiratory time is required to calculate I:E ratio, but this parameter is missing in the given dataset.")
        if "expiry-time" not in dataframe.columns:
            raise RuntimeError("Expiratory time is required to calculate I:E ratio, but this parameter is missing in the given dataset.")
        ie_values = []
        for i in dataframe["inspiry-time"].index:
            ie_values.append(dataframe["inspiry-time"][i] / dataframe["expiry-time"][i])
        dataframe["i-e"] = ie_values
        return dataframe
    
    def calculate_lymphocyte_percentage(self, dataframe):
        if "lymphocytes (relative)" in dataframe.columns:
            print("Skipping calculation of lymphocyte percentage since it already exists in the given dataset")
            return dataframe
        if "lymphocytes_abs" not in dataframe.columns:
            raise RuntimeError("absolute number of lymphocytes is required to calculate lymphocyte percentage, but this parameter is missing in the given dataset.")
        if "leucocytes" not in dataframe.columns:
            raise RuntimeError("Number of leucocytes is required to calculate lymphocyte percentage, but this parameter is missing in the given dataset.")
        lymphocyte_values = []
        for i in dataframe["lymphocytes_abs"].index:
            lymphocyte_values.append(dataframe["lymphocytes_abs"][i] / dataframe["leucocytes"][i])
        dataframe["lymphocytes (percentage)"] = lymphocyte_values
        return dataframe
    

    
    def add_param_to_calculate(self, param):
        if param in self.possible_params:
            if param not in self.params_to_calculate:
                self.params_to_calculate.append(param)
        else:
            raise RuntimeError("Parameter " + param + " is currently not supported/not known and cannot be calculated")
        
    def remove_param_to_calculate(self, param):
        if param in self.params_to_calculate:
            self.params_to_calculate.remove(param)

    def set_params_to_calculate(self, param_list):
        for param in param_list:
            if param not in self.possible_params:
                raise RuntimeError("Parameter " + param + " is currently not supported/ not known and cannot be calculated")
        self.params_to_calculate = param_list