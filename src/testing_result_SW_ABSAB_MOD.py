import json
import argparse
import logging
import os
from datetime import datetime
from multiprocessing import Pool

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
    LOG_DIR = "../Data/logs/prepare_all_data_DeepAnt"
    os.makedirs(LOG_DIR, exist_ok=True)

    log_filename = os.path.join(LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    data_frame = _load_numpy_file("../Data/uka_data_050623.npy")
    patient_ids = data_frame["patient_id"].unique().tolist()

    parser = argparse.ArgumentParser(description="Test SW_ABSAD_MOD")
    parser.add_argument("--job_id", type=int, help="Job ID, used to identify which patients to test")
    args = parser.parse_args()
    job_id = args.job_id
    start_patient_inx = job_id * 500
    end_patient_inx = start_patient_inx + 500
    relevant_ids = patient_ids[start_patient_inx:end_patient_inx]
    detector = SW_ABSAD_Mod_Detector(use_cl_modification=True, retrain_after_gap=True, variance_check=False, use_columns="", clean_training_window=True)
    patient_dfs = [data_frame[data_frame["patient_id"] == patient_id] for patient_id in relevant_ids]
    with Pool(processes=5) as pool:
        pool.starmap(detector.run, [(patient_dfs[i], i , len(patient_dfs)-1) for i in range(len(patient_dfs))])









