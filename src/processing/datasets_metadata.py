from typing import Union, Dict
import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AnomalyDetectionMetaData(BaseModel):
    """Metadata describing anomaly detection configuration and results"""
    name: str
    anomaly_detection_algorithm: str
    anomaly_handling_strategy: str
    anomaly_fixing_algorithm: Union[str, None] = None
    anomaly_threshold: Union[float, None] = None
    columns_checked: Union[list[dict[str, list[str]]], None] = None
    anomaly_statistics: dict[str, dict[str, Union[int, float]]]
    algorithm_specific_settings: dict[str, Union[str, int, float, bool, dict, list, None]] = None


class ImputationMetaData(BaseModel):
    """Metadata describing data imputation configuration"""
    impute_empty_cells: bool
    imputation_parameter_algorithm_dict: Union[Dict[str, str], None]


class ImputationMetaDataManager:
    """Manages merging of imputation metadata from multiple processing runs"""

    @staticmethod
    def merge(old: ImputationMetaData, new: ImputationMetaData) -> ImputationMetaData:
        """
        Merges two ImputationMetaData objects, combining information from old and new runs.
        
        Args:
            old: Existing imputation metadata
            new: New imputation metadata to merge
            
        Returns:
            Merged ImputationMetaData object
        """
        logger.debug("Merging imputation metadata...")
        
        if old and new:
            impute_empty_cells = old.impute_empty_cells or new.impute_empty_cells
            
            # Merge imputation algorithm dictionaries
            if not old.imputation_parameter_algorithm_dict and not new.imputation_parameter_algorithm_dict:
                imputation_parameter_algorithm_dict = None
            elif not old.imputation_parameter_algorithm_dict and new.imputation_parameter_algorithm_dict:
                imputation_parameter_algorithm_dict = new.imputation_parameter_algorithm_dict
            elif old.imputation_parameter_algorithm_dict and not new.imputation_parameter_algorithm_dict:
                imputation_parameter_algorithm_dict = old.imputation_parameter_algorithm_dict
            else:
                imputation_parameter_algorithm_dict = old.imputation_parameter_algorithm_dict
                if "all" not in imputation_parameter_algorithm_dict.keys():
                    for key in new.imputation_parameter_algorithm_dict.keys():
                        if key not in imputation_parameter_algorithm_dict.keys():
                            imputation_parameter_algorithm_dict[key] = old.imputation_parameter_algorithm_dict[key]
            
            logger.debug(f"Imputation metadata merged. Empty cell imputation: {impute_empty_cells}")
            return ImputationMetaData(
                impute_empty_cells=impute_empty_cells, 
                imputation_parameter_algorithm_dict=imputation_parameter_algorithm_dict
            )
        elif old and not new:
            logger.debug("Using existing imputation metadata (no new metadata to merge)")
            return old
        else:
            logger.debug("Using new imputation metadata (no existing metadata)")
            return new


class UnitConversionMetaData(BaseModel):
    """Metadata describing unit conversion operations applied to features"""
    conversions: Dict[str, str]


class ParamCalculationMetaData(BaseModel):
    """Metadata describing calculated parameters"""
    calculated_parameters: list


class ParamCalculationMetaDataManager:
    """Manages merging of parameter calculation metadata from multiple processing runs"""

    @staticmethod
    def merge(old: ParamCalculationMetaData, new: ParamCalculationMetaData) -> ParamCalculationMetaData:
        """
        Merges parameter calculation metadata from old and new processing runs.
        Combines lists of calculated parameters.
        
        Args:
            old: Existing parameter calculation metadata
            new: New parameter calculation metadata to merge
            
        Returns:
            Merged ParamCalculationMetaData object
        """
        logger.debug("Merging parameter calculation metadata...")
        
        if old and new:
            new_list = old.calculated_parameters + new.calculated_parameters
            logger.debug(f"Combined calculated parameters: {new_list}")
            return ParamCalculationMetaData(calculated_parameters=new_list)
        elif old:
            logger.debug("Using existing parameter calculation metadata")
            return old
        else:
            logger.debug("Using new parameter calculation metadata")
            return new


class OnsetDetectionMetaData(BaseModel):
    """Metadata describing ARDS onset detection configuration"""
    onset_detection_algorithm: str
    onset_detection_return_type: str
    series_begin: Union[float, None]
    series_end: Union[float, None]
    remove_ards_patients_without_onset: bool
    update_ards_diagnose: Union[bool, None]
    fill_cells_out_of_patient_stay: Union[bool, None]


class FilteringMetaData(BaseModel):
    """Metadata describing filtering operations applied to data"""
    applied_filters: Union[list, None]


class FilteringMetaDataManager:
    """Manages merging of filtering metadata from multiple processing runs"""

    @staticmethod
    def merge(old: FilteringMetaData, new: FilteringMetaData) -> FilteringMetaData:
        """
        Merges filtering metadata from old and new processing runs.
        Combines applied filters and removes conflicts (e.g., Lite and Strict filters).
        
        Args:
            old: Existing filtering metadata
            new: New filtering metadata to merge
            
        Returns:
            Merged FilteringMetaData object
        """
        logger.debug("Merging filtering metadata...")
        
        if old and new:
            new_list = old.applied_filters + new.applied_filters
            filter_set = set(new_list)
            
            # Remove conflicting filter types
            if "Lite" in filter_set and "Strict" in filter_set:
                logger.warning("Conflicting filters detected (Lite and Strict). Removing Lite filter.")
                filter_set.remove("Lite")
            
            filter_list = list(filter_set)
            logger.debug(f"Merged filters: {filter_list}")
            return FilteringMetaData(applied_filters=filter_list)
        elif old:
            logger.debug("Using existing filtering metadata")
            return old
        else:
            logger.debug("Using new filtering metadata")
            return new


class FeatureSelectionMetaData(BaseModel):
    """Metadata describing feature selection configuration and results"""
    feature_selection_algorithm: str
    min_required_variance: Union[float, None]
    num_features_to_select: Union[int, None]
    first_selection: bool


class FeatureSelectionMetaDataManager:
    """Manages merging of feature selection metadata from multiple processing runs"""

    @staticmethod
    def merge(old: FeatureSelectionMetaData, new: FeatureSelectionMetaData) -> FeatureSelectionMetaData:
        """
        Merges feature selection metadata from old and new processing runs.
        Uses the new metadata configuration as it represents the latest feature selection.
        
        Args:
            old: Existing feature selection metadata
            new: New feature selection metadata to merge
            
        Returns:
            Merged FeatureSelectionMetaData object with first_selection set to False
        """
        logger.debug("Merging feature selection metadata...")
        
        if old and new:
            first_selection = False
            feature_selection_algorithm = new.feature_selection_algorithm
            min_required_variance = new.min_required_variance
            num_features_to_select = new.num_features_to_select
            
            logger.debug(f"Using new feature selection config: {feature_selection_algorithm}. First selection: False")
            return FeatureSelectionMetaData(
                feature_selection_algorithm=feature_selection_algorithm,
                min_required_variance=min_required_variance,
                num_features_to_select=num_features_to_select,
                first_selection=first_selection
            )
        elif old:
            logger.debug("Using existing feature selection metadata")
            return old
        else:
            logger.debug("Using new feature selection metadata")
            return new


class TimeseriesMetaData(BaseModel):
    """
    Comprehensive metadata describing a timeseries dataset.
    Tracks datasource, processing history, ARDS percentage, and relationships to parent datasets.
    
    TODO: Validate that if is_child is True parent_meta_data_location exists
    TODO: Validate that if feature_selection first_selection is False is_child is True
    """
    datasource: str
    dataset_location: Union[str, None]
    dataset_type: str
    additional_information: Union[str, None]

    imputation: Union[ImputationMetaData, None]
    unit_conversion: Union[UnitConversionMetaData, None]
    parameter_calculation: Union[ParamCalculationMetaData, None]
    onset_detection: Union[OnsetDetectionMetaData, None]

    filtering: Union[FilteringMetaData, None]
    feature_selection: Union[FeatureSelectionMetaData, None]
    percentage_of_ards: float
    is_child: bool
    parent_meta_data_location: Union[str, None]


class TimeSeriesMetaDataManagement:
    """Manages creation, merging, and persistence of timeseries metadata objects"""

    @staticmethod
    def factory_method(processing_meta_data: dict, path: str, percentage_ards: float, dataset_type: str, 
                      additional_information: str = None) -> TimeseriesMetaData:
        """
        Creates a new TimeseriesMetaData object from processing metadata and dataset information.
        
        Args:
            processing_meta_data: Dictionary containing processing step metadata
            path: File path where the dataset will be stored
            percentage_ards: Percentage of ARDS samples in the dataset (0.0 to 1.0)
            dataset_type: Type of dataset (e.g., 'Training', 'Test', 'Complete')
            additional_information: Optional additional information about the dataset
            
        Returns:
            TimeseriesMetaData object with all processing history
        """
        logger.debug(f"Creating new metadata for {dataset_type} dataset")
        logger.debug(f"ARDS percentage: {percentage_ards*100:.2f}%, Path: {path}")

        meta_data_dataset = TimeseriesMetaData(
            datasource=processing_meta_data["database_name"],
            dataset_location=path,
            dataset_type=dataset_type,
            additional_information=additional_information,
            imputation=processing_meta_data["imputation"],
            unit_conversion=processing_meta_data["unit_conversion"],
            parameter_calculation=processing_meta_data["param_calculation"],
            onset_detection=processing_meta_data["onset_determination"],
            filtering=processing_meta_data["filtering"],
            feature_selection=processing_meta_data["feature_selection"],
            percentage_of_ards=percentage_ards,
            is_child=False,
            parent_meta_data_location=None
        )
        
        logger.debug("Metadata object created successfully")
        return meta_data_dataset

    @staticmethod
    def merge_meta_data(existing_meta_data: TimeseriesMetaData, new_meta_data: TimeseriesMetaData) -> TimeseriesMetaData:
        """
        Merges metadata from an existing dataset with new metadata.
        Used when creating train/test splits from a complete dataset.
        
        Args:
            existing_meta_data: Metadata from the parent/complete dataset
            new_meta_data: Metadata from the new child dataset
            
        Returns:
            Merged TimeseriesMetaData object with is_child=True and parent reference
        """
        logger.debug("Merging existing and new timeseries metadata...")
        
        datasource = existing_meta_data.datasource
        dataset_location = new_meta_data.dataset_location
        dataset_type = new_meta_data.dataset_type
        
        # Merge additional information
        additional_information = existing_meta_data.additional_information + ", " + new_meta_data.additional_information
        logger.debug(f"Additional information: {additional_information}")

        # Merge imputation metadata
        if new_meta_data.imputation or existing_meta_data.imputation:
            imputation = ImputationMetaDataManager.merge(existing_meta_data.imputation, new_meta_data.imputation)
        else:
            imputation = None

        unit_conversion = existing_meta_data.unit_conversion

        # Merge parameter calculation metadata
        if existing_meta_data.parameter_calculation or new_meta_data.parameter_calculation:
            parameter_calculation = ParamCalculationMetaDataManager.merge(
                existing_meta_data.parameter_calculation, 
                new_meta_data.parameter_calculation
            )
        else:
            parameter_calculation = None
            
        onset_detection = existing_meta_data.onset_detection

        # Merge filtering metadata
        if existing_meta_data.filtering or new_meta_data.filtering:
            filtering = FilteringMetaDataManager.merge(existing_meta_data.filtering, new_meta_data.filtering)
        else:
            filtering = None

        # Merge feature selection metadata
        if existing_meta_data.feature_selection or new_meta_data.feature_selection:
            feature_selection = FeatureSelectionMetaDataManager.merge(
                existing_meta_data.feature_selection, 
                new_meta_data.feature_selection
            )
        else:
            feature_selection = None

        percentage_of_ards = existing_meta_data.percentage_of_ards
        is_child = True
        parent_meta_data_location = existing_meta_data.dataset_location
        
        logger.debug(f"Metadata merged. Child dataset references parent at: {parent_meta_data_location}")

        return TimeseriesMetaData(
            datasource=datasource,
            dataset_location=dataset_location,
            dataset_type=dataset_type,
            additional_information=additional_information,
            imputation=imputation,
            unit_conversion=unit_conversion,
            parameter_calculation=parameter_calculation,
            onset_detection=onset_detection,
            filtering=filtering,
            feature_selection=feature_selection,
            percentage_of_ards=percentage_of_ards,
            is_child=is_child,
            parent_meta_data_location=parent_meta_data_location
        )

    @staticmethod
    def extract_procesing_meta_data(meta_data_dataset: TimeseriesMetaData) -> dict:
        """
        Extracts processing metadata from a TimeseriesMetaData object into a flat dictionary.
        
        Args:
            meta_data_dataset: TimeseriesMetaData object to extract from
            
        Returns:
            Dictionary containing extracted processing metadata
        """
        logger.debug("Extracting processing metadata...")
        
        processing_meta_data = {
            "database_name": meta_data_dataset.datasource,
            "imputation": meta_data_dataset.imputation,
            "unit_conversion": meta_data_dataset.unit_conversion,
            "param_calculation": meta_data_dataset.parameter_calculation,
            "onset_determination": meta_data_dataset.onset_detection,
            "filtering": meta_data_dataset.filtering,
            "feature_selection": meta_data_dataset.feature_selection
        }
        
        logger.debug("Processing metadata extracted successfully")
        return processing_meta_data

    @staticmethod
    def write(meta_data_dataset: TimeseriesMetaData):
        """
        Persists TimeseriesMetaData to a JSON file.
        The file is saved alongside the dataset with '_meta_data.json' suffix.
        
        Args:
            meta_data_dataset: TimeseriesMetaData object to persist
        """
        path = meta_data_dataset.dataset_location + "_meta_data.json"
        logger.info(f"Writing metadata to: {path}")
        
        try:
            with open(path, "w") as f:
                f.write(meta_data_dataset.model_dump_json(indent=4))
            logger.info(f"Metadata successfully written to {path}")
        except Exception as e:
            logger.error(f"Failed to write metadata to {path}: {str(e)}")
            raise

    @staticmethod
    def load_from_dict(meta_data_dict: dict) -> TimeseriesMetaData:
        """
        Reconstructs a TimeseriesMetaData object from a dictionary.
        Typically loaded from JSON file. Handles conversion of nested metadata objects.
        
        Args:
            meta_data_dict: Dictionary containing metadata (typically from JSON)
            
        Returns:
            Reconstructed TimeseriesMetaData object
        """
        logger.debug(f"Loading metadata from dictionary for dataset: {meta_data_dict.get('dataset_type', 'unknown')}")
        
        content = {
            "datasource": meta_data_dict["datasource"],
            "dataset_location": meta_data_dict["dataset_location"],
            "dataset_type": meta_data_dict["dataset_type"],
            "additional_information": meta_data_dict["additional_information"],
            "imputation": ImputationMetaData(**meta_data_dict["imputation"]) if meta_data_dict["imputation"] else None,
            "unit_conversion": UnitConversionMetaData(**meta_data_dict["unit_conversion"]) if meta_data_dict["unit_conversion"] else None,
            "parameter_calculation": ParamCalculationMetaData(**meta_data_dict["parameter_calculation"]) if meta_data_dict["parameter_calculation"] else None,
            "onset_detection": OnsetDetectionMetaData(**meta_data_dict["onset_detection"]) if meta_data_dict["onset_detection"] else None,
            "filtering": FilteringMetaData(**meta_data_dict["filtering"]) if meta_data_dict["filtering"] else None,
            "feature_selection": FeatureSelectionMetaData(**meta_data_dict["feature_selection"]) if meta_data_dict["feature_selection"] else None,
            "percentage_of_ards": meta_data_dict["percentage_of_ards"],
            "is_child": meta_data_dict["is_child"],
            "parent_meta_data_location": meta_data_dict["parent_meta_data_location"]
        }
        
        logger.debug(f"Metadata loaded successfully. Is child: {content['is_child']}")
        return TimeseriesMetaData(**content)


