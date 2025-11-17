from processing.datasets_metadata import TimeseriesMetaData, TimeSeriesMetaDataManagement

from pydantic import BaseModel
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TimeSeriesDataset(BaseModel):
    """
    Represents a timeseries dataset with its associated metadata.
    Contains the actual data content and all metadata about processing steps and dataset characteristics.
    """
    class Config:
        arbitrary_types_allowed = True

    content: pd.DataFrame
    """DataFrame containing the actual timeseries data"""
    
    meta_data: TimeseriesMetaData
    """Metadata object describing dataset characteristics and processing history"""


class TimeSeriesDatasetManagement:
    """
    Manages creation, persistence, and metadata handling for timeseries datasets.
    Handles both creation of new datasets and merging with existing metadata.
    """
    
    @staticmethod
    def factory_method(dataset: pd.DataFrame, processing_meta_data: dict, path: str, dataset_type: str, 
                      additional_information: str = None, existing_meta_data: TimeseriesMetaData = None) -> TimeSeriesDataset:
        """
        Factory method to create TimeSeriesDataset instances with appropriate metadata.
        Routes to either existing metadata merge or new metadata creation.
        
        Args:
            dataset: DataFrame containing the timeseries data
            processing_meta_data: Dictionary with metadata from preprocessing steps
            path: File path for saving the dataset
            dataset_type: Type of dataset (e.g., 'Training', 'Test', 'Complete')
            additional_information: Optional additional information to include in metadata
            existing_meta_data: Optional existing metadata to merge with new metadata
            
        Returns:
            TimeSeriesDataset object with data and associated metadata
        """
        logger.info(f"Creating TimeSeriesDataset (type: {dataset_type})...")
        logger.debug(f"Dataset shape: {dataset.shape}, Path: {path}")
        
        if existing_meta_data:
            logger.info(f"Creating dataset with merged metadata for {dataset_type} set")
            return TimeSeriesDatasetManagement._factory_existing_meta_data(
                dataset, processing_meta_data, path, dataset_type, additional_information, existing_meta_data
            )
        else:
            logger.info(f"Creating dataset with new metadata for {dataset_type} set")
            return TimeSeriesDatasetManagement._factory_new_meta_data(
                dataset, processing_meta_data, path, dataset_type, additional_information
            )

    @staticmethod
    def _factory_existing_meta_data(dataset: pd.DataFrame, processing_meta_data: dict, path: str, dataset_type: str, 
                                   additional_information: str = None, existing_meta_data: TimeseriesMetaData = None) -> TimeSeriesDataset:
        """
        Creates a TimeSeriesDataset by merging new metadata with existing metadata.
        Used when processing steps have already been applied and metadata exists.
        
        Args:
            dataset: DataFrame containing the timeseries data
            processing_meta_data: Dictionary with metadata from preprocessing steps
            path: File path for saving the dataset
            dataset_type: Type of dataset (e.g., 'Training', 'Test', 'Complete')
            additional_information: Optional additional information
            existing_meta_data: Existing metadata to merge with
            
        Returns:
            TimeSeriesDataset with merged metadata
        """
        logger.debug(f"Computing ARDS percentage for {dataset_type} dataset...")
        new_percentage_ards = dataset["ards"].sum() / len(dataset.index)
        logger.info(f"ARDS percentage in {dataset_type} set: {new_percentage_ards*100:.2f}%")
        
        logger.debug("Creating new metadata...")
        new_meta_data = TimeSeriesMetaDataManagement.factory_method(
            processing_meta_data, path, new_percentage_ards, dataset_type, additional_information
        )
        
        logger.debug("Merging existing and new metadata...")
        merged_meta_data = TimeSeriesMetaDataManagement.merge_meta_data(existing_meta_data, new_meta_data)
        logger.info(f"Successfully merged metadata for {dataset_type} dataset")
        
        return TimeSeriesDataset(content=dataset, meta_data=merged_meta_data)

    @staticmethod
    def _factory_new_meta_data(dataset: pd.DataFrame, processing_meta_data: dict, path: str, dataset_type: str, 
                              additional_information: str = None) -> TimeSeriesDataset:
        """
        Creates a TimeSeriesDataset with fresh metadata.
        Used when creating a dataset for the first time or without existing metadata.
        
        Args:
            dataset: DataFrame containing the timeseries data
            processing_meta_data: Dictionary with metadata from preprocessing steps
            path: File path for saving the dataset
            dataset_type: Type of dataset (e.g., 'Training', 'Test', 'Complete')
            additional_information: Optional additional information
            
        Returns:
            TimeSeriesDataset with new metadata
        """
        logger.debug(f"Computing ARDS percentage for {dataset_type} dataset...")
        percentage_ards = dataset["ards"].sum() / len(dataset.index)
        logger.info(f"ARDS percentage in {dataset_type} set: {percentage_ards*100:.2f}%")
        logger.info(f"Total samples in {dataset_type} set: {len(dataset.index)}")
        
        logger.debug("Creating new metadata object...")
        meta_data = TimeSeriesMetaDataManagement.factory_method(
            processing_meta_data, path, percentage_ards, dataset_type, additional_information
        )
        logger.info(f"Successfully created metadata for {dataset_type} dataset")
        
        return TimeSeriesDataset(content=dataset, meta_data=meta_data)

    @staticmethod
    def write(timeseries_dataset: TimeSeriesDataset):
        """
        Persists a TimeSeriesDataset to disk.
        Saves both the data content as CSV and the associated metadata as JSON.
        
        Args:
            timeseries_dataset: TimeSeriesDataset object to persist
        """
        dataset_type = timeseries_dataset.meta_data.dataset_type if hasattr(timeseries_dataset.meta_data, 'dataset_type') else "unknown"
        logger.info(f"Writing {dataset_type} dataset to disk...")
        
        path = timeseries_dataset.meta_data.dataset_location + ".csv"
        logger.info(f"Saving dataset content to: {path}")
        logger.debug(f"Dataset shape: {timeseries_dataset.content.shape}")
        
        try:
            timeseries_dataset.content.to_csv(path, index=False, header=True)
            logger.info(f"Dataset successfully saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save dataset to {path}: {str(e)}")
            raise
        
        logger.debug("Writing associated metadata...")
        try:
            TimeSeriesMetaDataManagement.write(timeseries_dataset.meta_data)
            logger.info(f"Metadata successfully written for {dataset_type} dataset")
        except Exception as e:
            logger.error(f"Failed to write metadata: {str(e)}")
            raise
