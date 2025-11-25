import math
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def prepare_multiprocessing(dataframe: pd.DataFrame, patients_per_process: int, max_processes: int) -> (list[pd.DataFrame], int):
    """
    Prepares data for multiprocessing by splitting the DataFrame into chunks based on patient IDs.
    Each chunk contains complete patient data (all records from first to last timestamp).

    Args:
        dataframe: Input DataFrame containing patient timeseries data with 'patient_id' and 'time' columns
        patients_per_process: Number of patients to process per job
        max_processes: Maximum allowed parallel processes

    Returns:
        Tuple containing:
            - list[pd.DataFrame]: List of DataFrames, each containing a subset of patients
            - int: Number of jobs (chunks) created
    """
    logger.info("Preparing data for multiprocessing...")
    if dataframe is None or dataframe.empty:
        logger.warning("Input dataframe is empty")
        return [], 0

    # preserve original dataframe index and order
    df = dataframe
    grouped = df.groupby("patient_id", sort=False)
    patient_ids = list(grouped.groups.keys())
    num_patients = len(patient_ids)
    logger.info(f"Total number of patients: {num_patients}")
    logger.info(f"Requested patients per process: {patients_per_process}")

    if patients_per_process <= 0:
        logger.warning("patients_per_process <= 0, defaulting to 1")
        patients_per_process = 1

    # Calculate initial number of jobs needed
    n_jobs = math.ceil(num_patients / patients_per_process) if num_patients > 0 else 0
    logger.info(f"Initial number of jobs to create: {n_jobs}")

    # Cap to max_processes and recompute patients_per_process so all patients are covered
    if n_jobs > max_processes:
        logger.warning(f"Calculated number of jobs ({n_jobs}) exceeds max processes ({max_processes}). Capping to max processes.")
        n_jobs = max_processes
        # Recalculate patients_per_process to evenly distribute patients across the capped number of jobs
        patients_per_process = math.ceil(num_patients / n_jobs) if n_jobs > 0 else num_patients
        logger.info(f"Recalculated patients_per_process to {patients_per_process} to use {n_jobs} jobs")

    # Build dictionary mapping each patient to their min and max positional indices (use positional iloc indices)
    logger.debug("Building patient time/position dictionary...")
    patient_pos_dict = {}
    for pid, idx in grouped.groups.items():
        # idx is an Index of labels; map labels to positional integer locations
        min_label = int(idx.min())
        max_label = int(idx.max())
        min_pos = df.index.get_indexer([min_label])[0]
        max_pos = df.index.get_indexer([max_label])[0]
        patient_pos_dict[pid] = (min_pos, max_pos)

    logger.debug(f"Patient position dictionary created with {len(patient_pos_dict)} entries")

    # Split data into chunks using recalculated patients_per_process (use positional slicing with iloc)
    logger.info("Splitting data into chunks...")
    process_pool_data_list = []

    for start_idx in range(0, num_patients, patients_per_process):
        end_idx = min(start_idx + patients_per_process, num_patients)
        first_patient = patient_ids[start_idx]
        last_patient = patient_ids[end_idx - 1]

        first_patient_begin_pos = patient_pos_dict[first_patient][0]
        last_patient_end_pos = patient_pos_dict[last_patient][1]

        # Use .iloc to slice by positional indices (inclusive of last)
        split_dataframe = df.iloc[first_patient_begin_pos:last_patient_end_pos + 1].copy()
        process_pool_data_list.append(split_dataframe)

        chunk_patient_count = split_dataframe["patient_id"].nunique()
        chunk_row_count = len(split_dataframe)
        logger.debug(f"Chunk covering patients {start_idx + 1}-{end_idx}: {chunk_patient_count} patients, {chunk_row_count} rows")

    # final number of jobs is length of generated list (should equal n_jobs)
    actual_n_jobs = len(process_pool_data_list)
    if actual_n_jobs != n_jobs:
        logger.debug(f"Adjusted n_jobs from {n_jobs} to actual created chunks {actual_n_jobs}")
        n_jobs = actual_n_jobs

    logger.info(f"Data preparation completed. Created {n_jobs} chunks for multiprocessing.")
    return process_pool_data_list, n_jobs


def get_processing_meta_data(database_name: str, processing_step_dict: dict) -> dict:
    """
    Collects metadata from all processing steps into a single dictionary.

    Args:
        database_name: Name of the database/datasource (e.g., 'MIMIC', 'UKA')
        processing_step_dict: Dictionary mapping processing step names to their processor objects
            Expected keys: 'filtering', 'unit_conversion', 'imputation', 'param_calculation',
                          'onset_determination', 'anomaly_detection'

    Returns:
        Dictionary containing aggregated metadata from all processing steps
    """
    logger.info("Collecting metadata from all processing steps...")
    meta_data_dict = {
        "database_name": database_name
    }

    for key, value in processing_step_dict.items():
        logger.debug(f"Collecting metadata for: {key}")
        if hasattr(value, 'meta_data'):
            meta_data_dict[key] = value.meta_data
            logger.debug(f"Metadata collected for {key}: {type(value.meta_data).__name__}")
        else:
            logger.warning(f"No metadata attribute found for processing step: {key}")
            meta_data_dict[key] = None

    logger.info(f"Metadata collection completed. Keys collected: {list(meta_data_dict.keys())}")
    return meta_data_dict

