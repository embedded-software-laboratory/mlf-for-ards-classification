from importlib.metadata import metadata
from pathlib import Path

import numpy as np
import os
import pandas as pd
from pydantic import ValidationError

from processing.datasets_metadata import  TimeseriesMetaData


class FileLoader:

    def load_file(self, file_path: str) -> (pd.DataFrame, TimeseriesMetaData):
        splitted_path = os.path.splitext(file_path)
        if splitted_path[1] == ".csv":
            dataset = self.load_csv_file(file_path)
        elif splitted_path[1] == ".npy":
            dataset = self.load_numpy_file(file_path)
        else:
            raise RuntimeError("File type " + splitted_path[1] + " not supported!")

        meta_data_file_path = splitted_path[0] + "_meta_data.json"
        meta_data_file = Path(meta_data_file_path)
        dataset_metadata = None
        if meta_data_file.is_file():
            file_content = meta_data_file.read_text()
            try:
                dataset_metadata = TimeseriesMetaData.model_validate_json(file_content)
            except ValidationError as err:
                dataset_metadata = None
                print(f"Error reading dataset metadata: {err}")

        return dataset, dataset_metadata
    @staticmethod
    def load_csv_file(file_path) -> pd.DataFrame:
        return pd.read_csv(file_path)

    @staticmethod
    def load_numpy_file(file_path) -> pd.DataFrame:
        data = np.load(file_path, mmap_mode="r+")
        variables_file = open(file_path + ".vars")
        variables = variables_file.read().split(" ")
        dataframe = pd.DataFrame(data, columns=variables)
        return dataframe
