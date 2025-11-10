
import logging
import os
import random

import pandas as pd
import torch

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """
    Set the random seed for reproducibility.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed: {seed}")

def check_device() -> str:
    """
    Check if a GPU is available and set the device accordingly.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    return device

def check_directory(directory: str):
    """
    Check if the directory exists, if not, create it.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Directory {directory} created.")
    else:
        logger.info(f"Directory {directory} already exists.")

def split_patients(dataframe: pd.DataFrame, anomaly_df: pd.DataFrame, patient_id: float) -> (pd.DataFrame, pd.DataFrame):
    """
        Splits the anomaly DataFrame and the DataFrame containing the original data for a specific patient_id from the rest of the data.

        Args:
            dataframe (pd.DataFrame): The original DataFrame containing the data.
            anomaly_df (pd.DataFrame): The DataFrame containing the anomalies.
            patient_id (float): The patient ID to filter the DataFrames by.

        Returns:
            tuple [pd.DataFrame, pd.DataFrame]: A tuple containing the DataFrame for the specific patient_id and the DataFrame with anomalies for that patient_id.

    """
    logger.info(f"Splitting data for patient_id: {patient_id}")
    patient_df = dataframe[dataframe["patient_id"] == patient_id].reset_index(drop=True)
    anomaly_df = anomaly_df[anomaly_df["patient_id"] == patient_id].reset_index(drop=True)
    return patient_df, anomaly_df
