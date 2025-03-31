import json
import argparse
import numpy as np
import pandas as pd

from processing.ad_algorithms.SW_ABSAD_MOD_Detector import SW_ABSAD_Mod_Detector


def _load_numpy_file(file_path) -> pd.DataFrame:
    data = np.load(file_path, mmap_mode="r+")
    variables_file = open(file_path + ".vars")
    variables = variables_file.read().split(" ")
    dataframe = pd.DataFrame(data, columns=variables)
    return dataframe


if __name__ == "__main__":
    data_frame = _load_numpy_file("C:/Users/Haenfry/Documents/Hiwi/MLP-Pipeline/mlp-framework/Data/uka_data_050623.npy")
    patient_ids = data_frame["patient_id"].unique().tolist()

    parser = argparse.ArgumentParser(description="Test SW_ABSAD_MOD")
    parser.add_argument("--job_id", type=int, help="Job ID, used to identify which patients to test")
    args = parser.parse_args()
    job_id = args.job_id
    start_patient_inx = job_id * 500
    end_patient_inx = start_patient_inx + 500
    relevant_ids = patient_ids[start_patient_inx:end_patient_inx]
    detector = SW_ABSAD_Mod_Detector(use_cl_modification=True, retrain_after_gap=True, variance_check=False, use_columns="", clean_training_window=True)
    for patient_id in relevant_ids:
        print(patient_id)
        patient_dataframe = data_frame[data_frame["patient_id"] == patient_id]
        # Assuming detector is defined and has a _predict method
        detector.run(patient_dataframe, 0, 1)









