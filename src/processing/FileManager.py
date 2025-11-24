from pathlib import Path
import numpy as np
import os
import pandas as pd
import logging
import glob
from pydantic import ValidationError
from processing.datasets_metadata import TimeseriesMetaData

logger = logging.getLogger(__name__)

class DataFileManager:
    """
    Manages loading of data files and their associated metadata.
    Supports loading CSV, NumPy, and pickle files, along with their metadata.
    """

    def load_file(self, file_path: str, import_type: str) -> (pd.DataFrame, TimeseriesMetaData):
        """
        Loads a data file and its associated metadata.
        
        Args:
            file_path: Path to the data file to be loaded
            import_type: Type of the data file (e.g., csv, numpy, pkl, extracted)
            
        Returns:
            Tuple containing the loaded DataFrame and its associated TimeseriesMetaData
        """
        logger.info(f"Loading file: {file_path}")
        splitted_path = os.path.splitext(file_path)
        
        # Load dataset based on file extension
        if import_type == "csv":
            logger.debug("Detected CSV file format.")
            dataset = self._load_csv_file(file_path)
        elif import_type == "numpy":
            logger.debug("Detected NumPy file format.")
            dataset = self._load_numpy_file(file_path)
        elif import_type == "pkl":
            logger.debug("Detected Pickle file format.")
            dataset = self._load_pkl_file(file_path)
        elif import_type == "split":
            logger.debug("Detected split data format. Loading directly as DataFrame.")
            dataset = self._load_split_file(file_path)
        else:
            logger.error(f"Unsupported file format: {import_type}")
            raise RuntimeError(f"Unsupported file format: {import_type}")

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

    @staticmethod
    def _load_split_file(file_path) -> pd.DataFrame:
        """
        Loads a split data file directly into a DataFrame.
        
        Args:
            file_path: Path to the split data file
            
        Returns:
            DataFrame containing the loaded data
        """
        logger.debug(f"Loading split data file: {file_path}")
        logger.info(f"Loading patient characteristics data")
        folder = os.path.join(file_path, "patient_characteristics")
        files = glob.glob(os.path.join(folder, "*.csv")) + glob.glob(os.path.join(folder, "*.parquet"))
        if not files:
            logger.warning(f"No files found in {folder}")
            return pd.DataFrame()
        dfs = []
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext == ".csv":
                dfs.append(pd.read_csv(f))
            elif ext == ".parquet":
                dfs.append(pd.read_parquet(f))
            else:
                logger.warning(f"Unsupported file format: {f}")
        if dfs:
            demographic_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(demographic_df)} rows and {len(demographic_df.columns)} columns from patient_characteristics")
            logger.debug(f"Columns in patient_characteristics: {demographic_df.columns.tolist()}")
        logger.info(f"Number of unique patients in demographic data: {demographic_df['identifier'].nunique()}")

        logger.info(f"Loading clinical measurements data")
        folder = os.path.join(file_path, "clinical_measurements")
        files = glob.glob(os.path.join(folder, "*.csv")) + glob.glob(os.path.join(folder, "*.parquet"))
        if not files:
            logger.warning(f"No files found in {folder}")
            return pd.DataFrame()
        dfs = []
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext == ".csv":
                dfs.append(pd.read_csv(f))
            elif ext == ".parquet":
                dfs.append(pd.read_parquet(f))
            else:
                logger.warning(f"Unsupported file format: {f}")
        if dfs:
            measurements_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Loaded {len(measurements_df)} rows and {len(measurements_df.columns)} columns from clinical_measurements")
            logger.debug(f"Columns in clinical_measurements: {measurements_df.columns.tolist()}")
        logger.info(f"Number of unique patients in vital data: {measurements_df['identifier'].nunique()}")
        logger.info(f"Combining patient characteristics and measurements")
        combined_df = pd.merge(measurements_df, demographic_df, on="identifier", how="left")
        return combined_df
    
