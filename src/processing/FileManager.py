rom pathlib import Path
import numpy as np
import os
import pandas as pd
import logging
import glob
from pydantic import ValidationError
from processing.datasets_metadata import TimeseriesMetaData
import re

logger = logging.getLogger(__name__)

class DataFileManager:
    """
    Manages loading of data files and their associated metadata.
    Supports loading CSV, NumPy, and pickle files, along with their metadata.
    """

    # =========================================================
    # ==================== PUBLIC INTERFACE ====================
    # =========================================================

    def load_file(self, file_path: str, import_type: str) -> (pd.DataFrame, TimeseriesMetaData):
        """
        Loads a data file + metadata and applies:
        1) gender conversion
        2) invalid numeric cleanup
        3) dtype optimization
        """
        logger.info(f"Loading file: {file_path}")
        splitted_path = os.path.splitext(file_path)
        
        # Load base data
        if import_type == "csv":
            dataset = self._load_csv_file(file_path)
        elif import_type == "numpy":
            dataset = self._load_numpy_file(file_path)
        elif import_type == "pkl":
            dataset = self._load_pkl_file(file_path)
        elif import_type == "split":
            dataset = self._load_split_file(file_path)
            dataset = self._convert_gender(dataset)                     # step 1
            dataset = self._clean_invalid_numeric_entries(dataset)      # step 2
            dataset = self._check_and_transform_dtypes(dataset)         # step 3
        else:
            raise RuntimeError(f"Unsupported file format: {import_type}")

        # Load metadata
        meta_data_file_path = splitted_path[0] + "_meta_data.json"
        metadata = None
        
        if Path(meta_data_file_path).is_file():
            file_content = Path(meta_data_file_path).read_text()
            try:
                metadata = TimeseriesMetaData.model_validate_json(file_content)
            except ValidationError as err:
                logger.warning(f"Error reading dataset metadata: {err}")

        return dataset, metadata

    # =========================================================
    # ==================== GENDER CLEANUP ======================
    # =========================================================

    @staticmethod
    def _convert_gender(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert gender column 'M'/'F' → 1/0 and leave NaN untouched.
        """
        if df is None or df.empty:
            return df

        if "gender" not in df.columns:
            return df

        logger.info("Converting gender values (M/F → 1/0)")
        df["gender"] = df["gender"].map({"M": 1, "F": 0})
        try:
            df["gender"] = df["gender"].astype("Int8")
        except Exception as e:
            logger.warning(f"Failed to convert gender column: {e}")

        return df

    # =========================================================
    # =========== INVALID VALUE CLEANING ADDED HERE ===========
    # =========================================================

    def _clean_invalid_numeric_entries(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans invalid numeric-like entries before dtype conversion.
        Examples cleaned:
            "<1.0" → 1.0
            ">5" → 5.0
            "Kein Material" → NaN
            "Nicht messbar" → NaN
            "--" → NaN
        """

        if dataframe is None or dataframe.empty:
            return dataframe

        logger.info("Cleaning invalid numeric entries...")

        numeric_pattern = re.compile(r"[-+]?\d*[\.,]?\d+")
        trash_values = {
            "kein", "keine", "kein material", "nicht", "nicht messbar", "missing",
            "na", "nan", "null", "none", "spur", "trace", "-", "--", "n/a",
            "positive", "negativ", "unauffällig"
        }

        for col in dataframe.columns:
            if not pd.api.types.is_object_dtype(dataframe[col]):
                continue

            cleaned = []
            for raw in dataframe[col].astype(str):
                s = raw.strip().lower()

                # Trash → NaN
                if s in trash_values or s.isalpha():
                    logger.info(f"Removed non-numeric value '{raw}' in column '{col}'")
                    cleaned.append(np.nan)
                    continue

                # Extract number from "<1.0"
                m = numeric_pattern.search(s)
                if m:
                    cleaned.append(m.group().replace(",", "."))
                    continue

                # Fully invalid
                logger.info(f"Removed invalid numeric value '{raw}' in column '{col}'")
                cleaned.append(np.nan)

            dataframe[col] = pd.to_numeric(cleaned, errors="coerce")

        return dataframe

    # =========================================================
    # ==================== DTYPE OPTIMIZATION =================
    # =========================================================

    @staticmethod
    def _check_and_transform_dtypes(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Final dtype optimization after correcting all values.
        """
        logger.info("Optimizing DataFrame dtypes...")

        if dataframe is None or dataframe.empty:
            return dataframe

        # dtype chooser helper
        def _select_nullable_int_dtype(min_v: int, max_v: int) -> str:
            if min_v >= 0:
                if max_v <= 255: return "UInt8"
                if max_v <= 65535: return "UInt16"
                if max_v <= 4294967295: return "UInt32"
                return "UInt64"
            else:
                if -128 <= min_v <= 127: return "Int8"
                if -32768 <= min_v <= 32767: return "Int16"
                if -2147483648 <= max_v <= 2147483647: return "Int32"
                return "Int64"

        for col in dataframe.columns:
            ser = dataframe[col]

            if pd.api.types.is_float_dtype(ser):
                try:
                    dataframe[col] = ser.astype("Float32")
                except:
                    pass
                continue

            if pd.api.types.is_integer_dtype(ser):
                try:
                    min_v, max_v = int(ser.min()), int(ser.max())
                    dataframe[col] = ser.astype(_select_nullable_int_dtype(min_v, max_v))
                except:
                    pass
                continue

            if pd.api.types.is_object_dtype(ser):
                coerced = pd.to_numeric(ser, errors="coerce")
                if coerced.notna().sum() > 0.5 * len(ser):
                    dataframe[col] = coerced.astype("Float32")

        logger.info("Dtype optimization complete.")
        return dataframe

    # =========================================================
    # ================ LOADERS =====================
    # =========================================================

    @staticmethod
    def _load_csv_file(file_path) -> pd.DataFrame:
        return pd.read_csv(file_path)

    @staticmethod
    def _load_pkl_file(file_path) -> pd.DataFrame:
        return pd.read_pickle(file_path)

    @staticmethod
    def _load_numpy_file(file_path) -> pd.DataFrame:
        data = np.load(file_path, mmap_mode="r+")
        with open(file_path + ".vars") as f:
            vars = f.read().split(" ")
        return pd.DataFrame(data, columns=vars)
    
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