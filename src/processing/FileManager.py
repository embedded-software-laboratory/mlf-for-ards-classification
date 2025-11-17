from pathlib import Path
import numpy as np
import os
import pandas as pd
import logging
from pydantic import ValidationError
from processing.datasets_metadata import TimeseriesMetaData

logger = logging.getLogger(__name__)

class DataFileManager:
    """
    Manages loading of data files and their associated metadata.
    Supports loading CSV, NumPy, and pickle files, along with their metadata.
    """

    def load_file(self, file_path: str) -> (pd.DataFrame, TimeseriesMetaData):
        """
        Loads a data file and its associated metadata.
        
        Args:
            file_path: Path to the data file to be loaded
            
        Returns:
            Tuple containing the loaded DataFrame and its associated TimeseriesMetaData
        """
        logger.info(f"Loading file: {file_path}")
        splitted_path = os.path.splitext(file_path)
        
        # Load dataset based on file extension
        if splitted_path[1] == ".csv":
            logger.debug("Detected CSV file format.")
            dataset = self._load_csv_file(file_path)
        elif splitted_path[1] == ".npy":
            logger.debug("Detected NumPy file format.")
            dataset = self._load_numpy_file(file_path)
        elif splitted_path[1] == ".pkl":
            logger.debug("Detected Pickle file format.")
            dataset = self._load_pkl_file(file_path)
        else:
            error_msg = f"File type {splitted_path[1]} not supported!"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Attempt to load associated metadata
        meta_data_file_path = splitted_path[0] + "_meta_data.json"
        meta_data_file = Path(meta_data_file_path)
        dataset_metadata = None
        
        if meta_data_file.is_file():
            logger.info(f"Loading metadata from: {meta_data_file_path}")
            file_content = meta_data_file.read_text()
            try:
                dataset_metadata = TimeseriesMetaData.model_validate_json(file_content)
                logger.info("Metadata loaded successfully.")
            except ValidationError as err:
                dataset_metadata = None
                logger.warning(f"Error reading dataset metadata: {err}")

        return dataset, dataset_metadata

    @staticmethod
    def _load_csv_file(file_path) -> pd.DataFrame:
        """
        Loads a CSV file into a DataFrame.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame containing the loaded data
        """
        logger.debug(f"Loading CSV file: {file_path}")
        return pd.read_csv(file_path)

    @staticmethod
    def _load_pkl_file(file_path) -> pd.DataFrame:
        """
        Loads a Pickle file into a DataFrame.
        
        Args:
            file_path: Path to the Pickle file
            
        Returns:
            DataFrame containing the loaded data
        """
        logger.debug(f"Loading Pickle file: {file_path}")
        return pd.read_pickle(file_path)

    @staticmethod
    def _load_numpy_file(file_path) -> pd.DataFrame:
        """
        Loads a NumPy file into a DataFrame.
        
        Args:
            file_path: Path to the NumPy file
            
        Returns:
            DataFrame containing the loaded data
        """
        logger.debug(f"Loading NumPy file: {file_path}")
        data = np.load(file_path, mmap_mode="r+")
        
        # Load variable names from associated .vars file
        variables_file_path = file_path + ".vars"
        logger.debug(f"Loading variable names from: {variables_file_path}")
        with open(variables_file_path) as variables_file:
            variables = variables_file.read().split(" ")
        
        dataframe = pd.DataFrame(data, columns=variables)
        logger.info("NumPy file loaded successfully.")
        return dataframe
