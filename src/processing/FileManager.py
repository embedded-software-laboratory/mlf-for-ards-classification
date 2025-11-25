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
            # After loading and merging, check and transform dtypes
            dataset = self._check_and_transform_dtypes(dataset)
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
    def _check_and_transform_dtypes(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize dtypes for memory and numeric operations.

        Strategy:
        - Convert 'gender' (M/F) to nullable Int8 with NaNs preserved.
        - Detect integer-like columns and downcast to the smallest
          suitable pandas nullable integer type (Int8/Int16/Int32/Int64 or UInt8/...).
        - Convert floating columns to pandas nullable Float32 where possible.
        - Try to coerce object columns that are mostly numeric before inference.
        - Leave non-numeric / string columns as-is (optionally could convert to category).
        """
        logger.info("Optimizing DataFrame dtypes for memory and numeric operations")

        if dataframe is None or dataframe.empty:
            logger.debug("DataFrame is empty; nothing to transform")
            return dataframe

        # 1) Handle gender specially: map 'M'/'F' -> 1/0, keep NaNs, use nullable Int8
        if 'gender' in dataframe.columns:
            logger.debug("Mapping 'gender' values to nullable Int8 (M->1, F->0)")
            dataframe['gender'] = dataframe['gender'].map({'M': 1, 'F': 0})
            try:
                dataframe['gender'] = dataframe['gender'].astype('Int8')
                logger.debug(f"  gender dtype -> {dataframe['gender'].dtype}")
            except Exception as e:
                logger.warning(f"  Failed to cast 'gender' to Int8: {e}")

        # helper: choose smallest nullable integer dtype for range
        def _select_nullable_int_dtype(min_v: int, max_v: int) -> str:
            # prefer unsigned if all values >= 0
            if min_v >= 0:
                if max_v <= 255:
                    return 'UInt8'
                if max_v <= 65535:
                    return 'UInt16'
                if max_v <= 4294967295:
                    return 'UInt32'
                return 'UInt64'
            else:
                if -128 <= min_v and max_v <= 127:
                    return 'Int8'
                if -32768 <= min_v and max_v <= 32767:
                    return 'Int16'
                if -2147483648 <= min_v and max_v <= 2147483647:
                    return 'Int32'
                return 'Int64'

        # Columns to skip (already handled)
        skip_cols = {'gender'}

        # 2) Iterate columns and infer best dtype
        for col in dataframe.columns:
            if col in skip_cols:
                continue

            ser = dataframe[col]

            # If object, attempt a safe numeric coercion if majority numeric
            if pd.api.types.is_object_dtype(ser):
                coerced = pd.to_numeric(ser, errors='coerce')
                non_null = coerced.notna().sum()
                if non_null > 0 and non_null / max(1, len(coerced)) >= 0.5:
                    # use coerced for inference but keep original values for assignment
                    ser_to_check = coerced
                    logger.debug(f"Column '{col}': object mostly numeric -> using coerced values for inference")
                else:
                    # leave non-numeric object column as-is
                    continue
            else:
                ser_to_check = ser

            # Drop NA for inference
            non_null_ser = ser_to_check.dropna()
            if non_null_ser.empty:
                logger.debug(f"Column '{col}': only NaNs, skipping dtype optimisation")
                continue

            # Check if integer-like: either integer dtype or floats with no fractional part
            is_int_like = False
            try:
                if pd.api.types.is_integer_dtype(non_null_ser):
                    is_int_like = True
                elif pd.api.types.is_float_dtype(non_null_ser):
                    arr = non_null_ser.to_numpy()
                    # fractional part check
                    frac = np.modf(arr)[0]
                    if np.all(frac == 0):
                        is_int_like = True
            except Exception:
                is_int_like = False

            if is_int_like:
                min_v = int(non_null_ser.min())
                max_v = int(non_null_ser.max())
                target_dtype = _select_nullable_int_dtype(min_v, max_v)
                try:
                    dataframe[col] = dataframe[col].astype(target_dtype)
                    logger.debug(f"Column '{col}' integer-like -> converted to {target_dtype}")
                except Exception as e:
                    logger.debug(f"Column '{col}' integer cast to {target_dtype} failed ({e}), leaving original dtype {dataframe[col].dtype}")
            else:
                # Not integer-like: try to downcast to pandas nullable Float32
                try:
                    dataframe[col] = dataframe[col].astype('Float32')
                    logger.debug(f"Column '{col}' converted to Float32")
                except Exception as e:
                    logger.debug(f"Column '{col}' Float32 cast failed ({e}), leaving dtype {dataframe[col].dtype}")

        logger.info("Dtype optimization complete")
        logger.debug(f"Resulting dtypes:\n{dataframe.dtypes}")
        return dataframe

    @staticmethod
    def _load_split_file(file_path) -> pd.DataFrame:
        """
        Loads a split data file directly into a DataFrame.
        
        Args:
            file_path: Path to the split data directory
            
        Returns:
            DataFrame containing the loaded data
        """
        logger.debug(f"Loading split data file: {file_path}")
        logger.info("Loading patient characteristics data")
        folder = os.path.join(file_path, "patient_characteristics/")
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

        logger.info("Loading clinical measurements data")
        folder = os.path.join(file_path, "clinical_measurements/")
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
        logger.info("Combining patient characteristics and measurements")
        combined_df = pd.merge(measurements_df, demographic_df, on="identifier", how="left")

        # Rename 'identifier' column to 'patient_id'
        if "identifier" in combined_df.columns:
            combined_df = combined_df.rename(columns={"identifier": "patient_id"})
            logger.info("Renamed column 'identifier' to 'patient_id'")
        else:
            logger.warning("Column 'identifier' not found in combined dataframe")

        # Rename 'timestamp' column to 'time'
        if "timestamp" in combined_df.columns:
            combined_df = combined_df.rename(columns={"timestamp": "time"})
            logger.info("Renamed column 'timestamp' to 'time'")
        else:
            logger.warning("Column 'timestamp' not found in combined dataframe")

        return combined_df