import math
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def prepare_multiprocessing(dataframe: pd.DataFrame, patients_per_process: int) -> (list[pd.DataFrame], int):
    """
    Prepares data for multiprocessing by splitting the DataFrame into chunks based on patient IDs.
    Each chunk contains complete patient data (all records from first to last timestamp).

    Args:
        dataframe: Input DataFrame containing patient timeseries data with 'patient_id' and 'time' columns
        patients_per_process: Number of patients to process per job

    Returns:
        Tuple containing:
            - list[pd.DataFrame]: List of DataFrames, each containing a subset of patients
            - int: Number of jobs (chunks) created
    """
    logger.info("Preparing data for multiprocessing...")
    patient_ids = dataframe["patient_id"].unique().tolist()
    num_patients = len(patient_ids)
    logger.info(f"Total number of patients: {num_patients}")
    logger.info(f"Patients per process: {patients_per_process}")

    # Calculate number of jobs needed
    n_jobs = math.ceil(num_patients / patients_per_process)
    logger.info(f"Number of jobs to create: {n_jobs}")

    # Build dictionary mapping each patient to their min and max time indices
    logger.debug("Building patient time range dictionary...")
    patient_pos_dict = {}
    patient_max_time_df = dataframe.groupby("patient_id")["time"].idxmax().reset_index(drop=True)
    patient_min_time_df = dataframe.groupby("patient_id")["time"].idxmin().reset_index(drop=True)

    for i in range(len(patient_ids)):
        patient_pos_dict[patient_ids[i]] = (int(patient_min_time_df[i]), int(patient_max_time_df[i]))

    logger.debug(f"Patient position dictionary created with {len(patient_pos_dict)} entries")

    # Split data into chunks
    logger.info("Splitting data into chunks...")
    process_pool_data_list = []
    index = 0

    for i in range(n_jobs):
        # Determine which patients belong to this job
        first_patient_idx = index + i * patients_per_process
        calculated_last_idx = index + (i + 1) * patients_per_process - 1
        last_patient_idx = calculated_last_idx if calculated_last_idx < num_patients else num_patients - 1

        first_patient = patient_ids[first_patient_idx]
        last_patient = patient_ids[last_patient_idx]

        # Get the time range for this chunk
        first_patient_begin_index = patient_pos_dict[first_patient][0]
        last_patient_end_index = patient_pos_dict[last_patient][1]

        # Extract the chunk
        split_dataframe = dataframe[first_patient_begin_index:last_patient_end_index]
        process_pool_data_list.append(split_dataframe)

        chunk_patient_count = len(split_dataframe["patient_id"].unique())
        chunk_row_count = len(split_dataframe)
        logger.debug(f"Job {i + 1}/{n_jobs}: {chunk_patient_count} patients, {chunk_row_count} rows")

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

