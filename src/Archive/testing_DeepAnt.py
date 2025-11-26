import logging
import os
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd


from processing.ad_algorithms.pytorch_DeepAnt import DeepAntDetector

def _load_numpy_file(file_path) -> pd.DataFrame:
    data = np.load(file_path, mmap_mode="r+")
    variables_file = open(file_path + ".vars")
    variables = variables_file.read().split(" ")
    dataframe = pd.DataFrame(data, columns=variables)
    return dataframe


LOG_DIR = "../Data/logs"
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

dataframe = _load_numpy_file("../Data/uka_data_050623.npy")
patient_ids = dataframe["patient_id"].unique().tolist()

#compliance_df = dataframe[["patient_id", "time", "compliance"]]
#compliance_df.dropna(inplace=True, subset=["compliance"])
#values_per_patient = compliance_df.groupby("patient_id").size()
#relevant_patients = values_per_patient[values_per_patient > 10].index.tolist()
#print(len(relevant_patients))
patients = len(patient_ids)

detector = DeepAntDetector(handling_strategy="delete_than_impute", fix_algorithm="interpolate")
patient_df_list = []
detector.run(dataframe, 0, 1)
