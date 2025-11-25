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
import gc

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
            import_type: Type of the data file (e.g., csv, numpy, pkl, extracted, split)
            
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
            logger.debug("Detected split data format.")
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
        df = pd.read_csv(file_path)
        logger.info(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")
        return df

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
        df = pd.read_pickle(file_path)
        logger.info(f"Pickle loaded: {len(df)} rows, {len(df.columns)} columns")
        return df

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
        logger.info(f"NumPy file loaded: {len(dataframe)} rows, {len(dataframe.columns)} columns")
        return dataframe

    @staticmethod
    def _load_split_file(file_path: str, csv_chunksize: int = 50_000) -> pd.DataFrame:
        """
        Memory-efficient loader für split data (demographics + clinical measurements).
        
        Strategie:
        1. Lädt demographics komplett (sollte klein sein)
        2. Liest clinical_measurements in Chunks
        3. Merged jeweils einen Chunk mit demographics
        4. Schreibt Chunks direkt zu Parquet (on-disk)
        5. Lädt final Parquet zurück (aber mit Speicher-Optimierungen)
        
        Args:
            file_path: Pfad zum split data Verzeichnis
            csv_chunksize: Anzahl Reihen pro Chunk beim CSV-Lesen
            
        Returns:
            Kombiniertes DataFrame (optimiert für Speicher)
        """
        logger.info(f"Loading split data (memory-efficient) from: {file_path}")
        
        # ===== SCHRITT 1: Demographics laden =====
        demog_folder = os.path.join(file_path, "patient_characteristics/")
        demog_files = glob.glob(os.path.join(demog_folder, "*.csv")) + \
                      glob.glob(os.path.join(demog_folder, "*.parquet"))
        
        if not demog_files:
            logger.error(f"No demographics files found in {demog_folder}")
            raise FileNotFoundError(f"No files found in {demog_folder}")

        logger.info(f"Found {len(demog_files)} demographics files")
        demog_dfs = []
        
        for f in demog_files:
            ext = os.path.splitext(f)[1].lower()
            logger.debug(f"Loading demographics from: {f}")
            try:
                if ext == ".csv":
                    df = pd.read_csv(f, dtype={'identifier': 'str'})
                elif ext == ".parquet":
                    df = pd.read_parquet(f)
                else:
                    logger.warning(f"Skipping unsupported format: {f}")
                    continue
                demog_dfs.append(df)
                logger.debug(f"  Loaded: {len(df)} rows")
            except Exception as e:
                logger.error(f"Failed to load demographics {f}: {e}")
                raise
        
        demographic_df = pd.concat(demog_dfs, ignore_index=True)
        logger.info(f"Demographics combined: {len(demographic_df)} rows, {len(demographic_df.columns)} columns")
        logger.debug(f"  Columns: {demographic_df.columns.tolist()}")
        
        # Speicher optimieren: identifier als category
        if "identifier" in demographic_df.columns:
            demographic_df["identifier"] = demographic_df["identifier"].astype("category")
            logger.debug("Set 'identifier' to category type (memory optimization)")
        
        # ===== SCHRITT 2: Clinical measurements in Chunks verarbeiten =====
        meas_folder = os.path.join(file_path, "clinical_measurements/")
        meas_files = glob.glob(os.path.join(meas_folder, "*.csv")) + \
                     glob.glob(os.path.join(meas_folder, "*.parquet"))
        
        if not meas_files:
            logger.error(f"No clinical measurements files found in {meas_folder}")
            raise FileNotFoundError(f"No files found in {meas_folder}")
        
        logger.info(f"Found {len(meas_files)} clinical measurement files")
        
        # Temporäre Parquet-Datei für on-disk Puffer
        temp_dir = tempfile.mkdtemp(prefix="split_combined_")
        combined_parquet_path = os.path.join(temp_dir, "combined.parquet")
        logger.info(f"Using temporary directory: {temp_dir}")
        
        writer = None
        total_rows = 0
        chunk_counter = 0
        
        # Merge-Spalte ermitteln
        merge_key = None
        if "identifier" in demographic_df.columns:
            merge_key = "identifier"
            logger.debug(f"Using '{merge_key}' as merge key")
        elif "patient_id" in demographic_df.columns:
            merge_key = "patient_id"
            logger.debug(f"Using '{merge_key}' as merge key")
        else:
            logger.warning("No common merge key found (identifier/patient_id). Will load without merge.")
        
        # ===== Iterate über clinical measurements =====
        for meas_file in meas_files:
            ext = os.path.splitext(meas_file)[1].lower()
            logger.info(f"Processing clinical measurements: {meas_file}")
            
            try:
                if ext == ".parquet":
                    logger.debug("  Reading parquet file in row groups...")
                    parquet_file = pq.ParquetFile(meas_file)
                    num_rg = parquet_file.num_row_groups
                    logger.debug(f"  File has {num_rg} row groups")
                    
                    for rg_idx in range(num_rg):
                        logger.debug(f"    Processing row group {rg_idx + 1}/{num_rg}...")
                        try:
                            table = parquet_file.read_row_group(rg_idx)
                            chunk_df = table.to_pandas()
                        except Exception as e:
                            logger.warning(f"    Row group read failed, retrying: {e}")
                            # Fallback: read gesamte Datei
                            chunk_df = pd.read_parquet(meas_file)
                        
                        # Merge mit demographics
                        if merge_key and merge_key in chunk_df.columns and merge_key in demographic_df.columns:
                            chunk_df[merge_key] = chunk_df[merge_key].astype(
                                demographic_df[merge_key].dtype
                            )
                            merged = chunk_df.merge(demographic_df, on=merge_key, how="left", 
                                                   suffixes=('', '_demog'))
                        else:
                            merged = chunk_df
                        
                        # Write zu Parquet
                        table_out = pa.Table.from_pandas(merged, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(combined_parquet_path, table_out.schema)
                            logger.debug(f"  Created ParquetWriter with schema")
                        writer.write_table(table_out)
                        
                        total_rows += len(merged)
                        chunk_counter += 1
                        logger.debug(f"    Chunk {chunk_counter} written ({len(merged)} rows)")
                        
                        # Speicher freigeben
                        del chunk_df, merged, table_out
                        gc.collect()
                
                elif ext == ".csv":
                    logger.debug(f"  Reading CSV in chunks of {csv_chunksize}...")
                    for chunk_idx, chunk_df in enumerate(
                        pd.read_csv(meas_file, chunksize=csv_chunksize, dtype={'identifier': 'str'})
                    ):
                        logger.debug(f"    Processing CSV chunk {chunk_idx + 1}...")
                        
                        # Merge mit demographics
                        if merge_key and merge_key in chunk_df.columns and merge_key in demographic_df.columns:
                            chunk_df[merge_key] = chunk_df[merge_key].astype(
                                demographic_df[merge_key].dtype
                            )
                            merged = chunk_df.merge(demographic_df, on=merge_key, how="left",
                                                   suffixes=('', '_demog'))
                        else:
                            merged = chunk_df
                        
                        # Write zu Parquet
                        table_out = pa.Table.from_pandas(merged, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(combined_parquet_path, table_out.schema)
                            logger.debug(f"  Created ParquetWriter with schema")
                        writer.write_table(table_out)
                        
                        total_rows += len(merged)
                        chunk_counter += 1
                        logger.debug(f"    Chunk {chunk_counter} written ({len(merged)} rows)")
                        
                        # Speicher freigeben
                        del chunk_df, merged, table_out
                        gc.collect()
                
                else:
                    logger.warning(f"Unsupported file format: {meas_file}")
                    continue
                    
            except Exception as e:
                logger.error(f"Failed to process {meas_file}: {e}")
                raise
        
        if writer is not None:
            writer.close()
            logger.info(f"ParquetWriter closed. Total written: {total_rows} rows across {chunk_counter} chunks")
        else:
            logger.error("No data was written to parquet file!")
            raise RuntimeError("No measurement data could be processed")
        
        # ===== SCHRITT 3: Parquet zurück zu DataFrame laden (mit Optimierungen) =====
        logger.info(f"Loading combined parquet into memory: {combined_parquet_path}")
        
        try:
            # Lese Parquet mit pyarrow für bessere Speicher-Kontrolle
            dataset = ds.dataset(combined_parquet_path, format="parquet")
            schema = dataset.schema
            logger.debug(f"Dataset schema: {schema}")
            
            # Lade in Batches, wenn möglich
            combined_df = dataset.to_table().to_pandas()
            
            logger.info(f"Final DataFrame loaded: {len(combined_df)} rows, {len(combined_df.columns)} columns")
            
            # Cleanup
            del demographic_df
            gc.collect()
            logger.debug("Cleaned up temporary objects")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Failed to load combined parquet: {e}")
            raise

