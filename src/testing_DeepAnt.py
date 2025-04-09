from multiprocessing import Pool

import numpy as np
import pandas as pd


from processing.ad_algorithms import DeepAntAnomalyDetector

def _load_numpy_file(file_path) -> pd.DataFrame:
    data = np.load(file_path, mmap_mode="r+")
    variables_file = open(file_path + ".vars")
    variables = variables_file.read().split(" ")
    dataframe = pd.DataFrame(data, columns=variables)
    return dataframe


dataframe = _load_numpy_file("../Data/uka_data_050623.npy")
patient_ids = dataframe["patient_id"].unique().tolist()

detector = DeepAntAnomalyDetector(handling_strategy="delete_than_impute", fix_algorithm="interpolate")
patient_df_list = []
detector.run(dataframe, 0, 1)
