import numpy as np
import os
import pandas as pd

class FileLoader():

    def load_file(self, file_path):
        splitted_path = os.path.splitext(file_path)
        if splitted_path[1] == ".csv":
            return self.load_csv_file(file_path)
        elif splitted_path[1] == ".npy":
            return self.load_numpy_file(file_path)
        else:
            raise RuntimeError("File type " + splitted_path[1] + " not supported!")
    @staticmethod
    def load_csv_file(file_path):
        return pd.read_csv(file_path)

    @staticmethod
    def load_numpy_file(file_path) -> pd.DataFrame:
        data = np.load(file_path, mmap_mode="r+")
        variables_file = open(file_path + ".vars")
        variables = variables_file.read().split(" ")
        dataframe = pd.DataFrame(data, columns=variables)
        return dataframe
