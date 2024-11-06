
import pandas as pd
from sklearn.model_selection import  train_test_split
import random
import math
class Data_segregator():

    def __init__(self, config):
        self.training_test_ratio = config["training_test_ratio"]
        self.ards_percentage = config["percentage_of_ards_patients"]
        self.seed = config["splitting_seed"]
    
    def segregate_data(self, dataframe):
        random.seed(self.seed)
        patientids = dataframe["patient_id"].to_list()
        num_patients = dataframe.shape[0]
        #first, create two datasets, one with the patients with and one with the patients without ards
        ards_mask = dataframe["ards"] == 1
        ards_data = dataframe[ards_mask]
        non_ards_data = dataframe[~ards_mask]

        num_ards_patients = ards_data.shape[0]
        num_non_ards_patients = non_ards_data.shape[0]

        if num_ards_patients > self.ards_percentage * num_patients:
            num_target_ards_patients = math.ceil(self.ards_percentage * num_patients)
            samples = random.sample(range(num_ards_patients), num_target_ards_patients)
            ards_data_sampled = ards_data.iloc[samples]
            data = pd.concat([ards_data_sampled, non_ards_data]).reset_index(drop=True)


        elif num_ards_patients < self.ards_percentage * num_patients:
            num_target_non_ards_patients = math.ceil((1-self.ards_percentage) * num_patients)
            samples = random.sample(range(num_non_ards_patients), num_target_non_ards_patients)
            non_ards_data_sampled = non_ards_data.iloc[samples]
            data = pd.concat([non_ards_data_sampled, ards_data]).reset_index(drop=True)
        else:
            data = pd.concat([ards_data, non_ards_data], axis=0).reset_index(drop=True)
        data.drop(["time", "patient_id"], axis=1, inplace=True)
        training_data, test_data = train_test_split(data, test_size=self.training_test_ratio, random_state=self.seed, shuffle=True, stratify=data["ards"])


        return training_data, test_data


    def set_training_test_ratio(self, new_ratio):
        self.training_test_ratio = new_ratio

    def set_ards_percentage(self, new_percentage):
        self.set_ards_percentage = new_percentage
    