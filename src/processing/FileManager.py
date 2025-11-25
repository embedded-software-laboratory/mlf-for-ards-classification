from pathlib import Path
import numpy as np
import os
import pandas as pd
import logging
import glob
from pydantic import ValidationError
from processing.datasets_metadata import TimeseriesMetaData
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

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
    def _load_split_file(file_path, return_parquet_path: bool = False, csv_chunksize: int = 100_000) -> pd.DataFrame:
        """
        Memory efficiently loads split data from a directory containing demographics and clinical measurements.
        """
        logger.info(f"Loading split data (memory-efficient) from: {file_path}")
        # --- load demographics (likely small) ---
        demog_folder = os.path.join(file_path, "patient_characteristics/")
        demog_files = glob.glob(os.path.join(demog_folder, "*.csv")) + glob.glob(os.path.join(demog_folder, "*.parquet"))
        if not demog_files:
            logger.warning(f"No files found in {demog_folder}")
            return pd.DataFrame()

        demog_dfs = []
        for f in demog_files:
            ext = os.path.splitext(f)[1].lower()
            if ext == ".csv":
                demog_dfs.append(pd.read_csv(f))
            elif ext == ".parquet":
                demog_dfs.append(pd.read_parquet(f))
        demographic_df = pd.concat(demog_dfs, ignore_index=True)
        logger.info(f"Loaded demographics: rows={len(demographic_df)}, cols={len(demographic_df.columns)}")
        logger.debug(f"Demographic columns: {demographic_df.columns.tolist()}")

        # reduce memory: set identifier to categorical if present
        if "identifier" in demographic_df.columns:
            demographic_df["identifier"] = demographic_df["identifier"].astype("category")

        # --- prepare on-disk output (parquet) ---
        temp_dir = tempfile.mkdtemp(prefix="split_combined_")
        combined_parquet_path = os.path.join(temp_dir, "combined.parquet")
        writer = None
        total_rows = 0

        # --- process clinical measurements in a memory-friendly way ---
        meas_folder = os.path.join(file_path, "clinical_measurements/")
        meas_files = glob.glob(os.path.join(meas_folder, "*.csv")) + glob.glob(os.path.join(meas_folder, "*.parquet"))
        if not meas_files:
            logger.warning(f"No files found in {meas_folder}")
            return pd.DataFrame()

        for f in meas_files:
            ext = os.path.splitext(f)[1].lower()
            logger.info(f"Processing measurements file {f}")
            if ext == ".parquet":
                # read parquet in row groups via pyarrow to limit memory (if large)
                try:
                    parquet_file = pq.ParquetFile(f)
                    for rg in range(parquet_file.num_row_groups):
                        table = parquet_file.read_row_group(rg)
                        chunk_df = table.to_pandas()
                        # merge chunk with demographics on 'identifier' if present
                        if "identifier" in chunk_df.columns and "identifier" in demographic_df.columns:
                            # ensure same dtype
                            chunk_df["identifier"] = chunk_df["identifier"].astype(demographic_df["identifier"].dtype)
                            merged = chunk_df.merge(demographic_df, on="identifier", how="left")
                        else:
                            merged = chunk_df
                        table_out = pa.Table.from_pandas(merged, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(combined_parquet_path, table_out.schema)
                        writer.write_table(table_out)
                        total_rows += len(merged)
                except Exception as e:
                    logger.warning(f"Parquet row-group read failed, falling back to pandas.read_parquet: {e}")
                    chunk_df = pd.read_parquet(f)
                    if "identifier" in chunk_df.columns and "identifier" in demographic_df.columns:
                        chunk_df["identifier"] = chunk_df["identifier"].astype(demographic_df["identifier"].dtype)
                        merged = chunk_df.merge(demographic_df, on="identifier", how="left")
                    else:
                        merged = chunk_df
                    table_out = pa.Table.from_pandas(merged, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(combined_parquet_path, table_out.schema)
                    writer.write_table(table_out)
                    total_rows += len(merged)

            elif ext == ".csv":
                # iterate CSV in chunks
                for chunk in pd.read_csv(f, chunksize=csv_chunksize):
                    if "identifier" in chunk.columns and "identifier" in demographic_df.columns:
                        # ensure same dtype to speed merge
                        chunk["identifier"] = chunk["identifier"].astype(demographic_df["identifier"].dtype)
                        merged = chunk.merge(demographic_df, on="identifier", how="left")
                    else:
                        merged = chunk
                    table_out = pa.Table.from_pandas(merged, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(combined_parquet_path, table_out.schema)
                    writer.write_table(table_out)
                    total_rows += len(merged)
            else:
                logger.warning(f"Unsupported file format: {f}. Skipping.")
                continue

        if writer is not None:
            writer.close()
        logger.info(f"Wrote combined dataset to on-disk parquet: {combined_parquet_path} (rows ~ {total_rows})")

        if return_parquet_path:
            # Return the parquet file path so caller can use out-of-core reading (pyarrow.dataset, dask, etc.)
            return combined_parquet_path

        # By default, return pandas DataFrame (note: will load full dataset into memory)
        logger.info("Loading combined parquet into memory as pandas.DataFrame (this will use memory proportional to dataset size)")
        dataset = ds.dataset(combined_parquet_path, format="parquet")
        combined_table = dataset.to_table()
        combined_df = combined_table.to_pandas()
        logger.info(f"Returning combined DataFrame with rows={len(combined_df)}")
        return combined_df

