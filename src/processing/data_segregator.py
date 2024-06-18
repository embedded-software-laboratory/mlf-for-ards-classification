import pandas as pd

class Data_segregator():

    def __init__(self, config):
        self.training_test_ratio = config["training_test_ratio"]
        self.ards_percentage = config["percentage_of_ards_patients"]
    
    def segregate_data(self, dataframe):
        patientids = dataframe["patient_id"].to_list()
        number_of_patients = len(set(patientids))
        #first, create two datasets, one with the patients with and one with the patients without ards
        ards_mask = dataframe["ards"] == 1
        ards_data = dataframe[ards_mask]
        non_ards_data = dataframe[~ards_mask]

        #calculate the current ards_ratio
        ards_patient_ids = set(ards_data["patient_id"].to_list())
        non_ards_patient_ids = set(non_ards_data["patient_id"].to_list())
        current_ards_ratio = len(set(ards_patient_ids))/number_of_patients

        #drop columns until the actual ards ratio maps to the desired one (if an ards ratio were set before)
        if self.ards_percentage != "None":
            if self.ards_percentage > current_ards_ratio:
                while self.ards_percentage > current_ards_ratio and len(non_ards_patient_ids) > 0:
                    removed_patient_id = non_ards_patient_ids.pop()
                    non_ards_data = non_ards_data.drop(non_ards_data[non_ards_data["patient_id"] == removed_patient_id].index)
                    number_of_patients-=1
                    current_ards_ratio = len(non_ards_patient_ids)/number_of_patients
            elif self.ards_percentage < current_ards_ratio:
                while self.ards_percentage < current_ards_ratio and len(ards_patient_ids) > 0:
                    removed_patient_id = ards_patient_ids.pop()
                    ards_data = ards_data.drop(ards_data[ards_data["patient_id"] == removed_patient_id].index)
                    number_of_patients-=1
                    current_ards_ratio = len(ards_patient_ids)/number_of_patients
        
        #split both datasets into test and training data
        ards_training_data = None
        ards_test_data = None
        non_ards_training_data = None
        non_ards_test_data = None

        ards_dataset_threshold = int(len(ards_patient_ids)*self.training_test_ratio)
        ards_training_subframes = []
        ards_test_subframes = []
        for i in range(len(ards_patient_ids)):
            if i < ards_dataset_threshold:
                ards_training_subframes.append((ards_data[ards_data["patient_id"] == list(ards_patient_ids)[i]]))
            else:
                ards_test_subframes.append((ards_data[ards_data["patient_id"] == list(ards_patient_ids)[i]]))
        
        if len(ards_training_subframes) > 0: ards_training_data = pd.concat(ards_training_subframes)
        if len(ards_test_subframes) > 0: ards_test_data = pd.concat(ards_test_subframes)
        
        non_ards_dataset_threshold = int(len(non_ards_patient_ids)*self.training_test_ratio)
        non_ards_training_subframes = []
        non_ards_test_subframes = []
        for i in range(len(non_ards_patient_ids)):
            if i < non_ards_dataset_threshold:
                non_ards_training_subframes.append((non_ards_data[non_ards_data["patient_id"] == list(non_ards_patient_ids)[i]]))
            else:
                non_ards_test_subframes.append((non_ards_data[non_ards_data["patient_id"] == list(non_ards_patient_ids)[i]]))
        
        if len(non_ards_training_subframes) > 0: non_ards_training_data = pd.concat(non_ards_training_subframes)
        if len(non_ards_test_subframes) > 0: non_ards_test_data = pd.concat(non_ards_test_subframes)

        training_data = pd.concat([ards_training_data, non_ards_training_data])
        test_data = pd.concat([ards_test_data, non_ards_test_data])

        return training_data, test_data


    def set_training_test_ratio(self, new_ratio):
        self.training_test_ratio = new_ratio

    def set_ards_percentage(self, new_percentage):
        self.set_ards_percentage = new_percentage
    