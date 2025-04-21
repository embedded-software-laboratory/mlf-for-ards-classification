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
dataframe = _load_numpy_file("/work/rwth1474/Data/time_series/uka_data_050623.npy")



detector = DeepAntDetector(handling_strategy="delete_than_impute", fix_algorithm="interpolate")
detector.prepare_full_data_for_storage(dataframe)
