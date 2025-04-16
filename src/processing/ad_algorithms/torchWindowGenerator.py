
import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple
import torch
import torch.utils.data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sympy import Union

logger = logging.getLogger(__name__)

class DataModule(torch.utils.data.Dataset):
    def __init__(self, data_x: np.ndarray, data_y: np.ndarray, device: str):
        """
            Pytorch Dataset for time series data.

            Each sample is a tuple (input_window, prediction_window)
            where:
                input_window: shape (num_features, input_width)
                prediction_window: shape (num_features, output_width)

            Args:
                data_x (np.ndarray): Input data of shape (num_samples, input_width, num_features).
                data_y (np.ndarray): Output data of shape (num_samples, output_width, num_features,).
                device (str): Device to use ('cpu' or 'cuda').
        """

        self.data_x = data_x
        self.data_y = data_y
        self.device = device


    def __len__(self) -> int:
        """
            Returns:
                int: Number of samples in the dataset.

        """
        return len(self.data_x)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        """
            Returns a sample from the dataset.

            Args:
                index (int): Index of the sample to retrieve.

            Returns:
                Tuple[np.ndarray, np.ndarray]: A tuple containing the input and output windows.
        """
        return (
            torch.tensor(self.data_x[index], device=self.device, dtype=torch.float32).transpose(0,1),
            torch.tensor(self.data_y[index], device=self.device, dtype=torch.float32).transpose(0,1)
        )

class WindowGenerator:

    def __init__(self, **kwargs ):
        """
            WindowGenerator for time series data.

            Args:
                input_width (int): The width of the input window.
                output_width (int): The width of the output window.
        """
        self.input_width = int(kwargs.get("input_width", 10))
        self.output_width = int(kwargs.get("output_width", 1))
        self.label_columns = list(kwargs.get("label_columns", []))
        self.features = list(kwargs.get("features", []))
        self.device = kwargs.get("device", "cpu")
        self.data_x = []
        self.data_y = []

    def split_data(self, data: pd.DataFrame) -> bool:
        """
            Splits the data into input and output windows.

            Args:
                data (pd.DataFrame): The input data.

            Returns:
                Tuple[np.ndarray, np.ndarray]: A tuple containing the input and output windows.
        """
        if self.input_width + self.output_width > len(data.index):

            logger.warning(f"The input and output windows are larger than the data length {len(data.index)}. Skipping...")
            return False

        for i in range(self.input_width, len(data.index) - self.output_width, self.output_width):
            x = data[self.features][i - self.input_width: i].to_numpy()
            y = data[self.label_columns][i+1:i+self.output_width].to_numpy()

            self.data_x.append(x)
            self.data_y.append(y)
        return True

    def generate_dataset(self) :
        if not self.data_x or not self.data_y:
            return None
        data_x = np.array(self.data_x, dtype=np.float32)
        data_y = np.array(self.data_y, dtype=np.float32)
        dataset = DataModule(data_x=data_x, data_y=data_y, device=self.device)
        self.data_x = []
        self.data_y = []
        return dataset










