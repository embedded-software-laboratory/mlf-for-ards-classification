import math
import pandas as pd


def prepare_multiprocessing(dataframe: pd.DataFrame, patients_per_process: int) -> (list[pd.DataFrame], int):
    patient_ids = list(dataframe["patient_id"].unique())
    num_patients = len(patient_ids)
    n_jobs = math.ceil(num_patients / patients_per_process)

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
        first_patient = patient_ids[index + i * patients_per_process]
        calculated_last_index = index + (i + 1) * patients_per_process - 1
        last_index = calculated_last_index if calculated_last_index < num_patients else num_patients - 1

        last_patient = patient_ids[last_index]
        first_patient_begin_index = patient_pos_dict[first_patient][0]
        last_patient_end_index = patient_pos_dict[last_patient][1]
        split_dataframe = dataframe[first_patient_begin_index:last_patient_end_index]
        process_pool_data_list.append(split_dataframe)

    return process_pool_data_list, n_jobs


def get_processing_meta_data(database_name: str, processing_step_dict: dict) -> dict:
    meta_data_dict = {
        "database_name": database_name
    }
    for key, value in processing_step_dict.items():
        meta_data_dict[key] = value.create_meta_data()
    return meta_data_dict

