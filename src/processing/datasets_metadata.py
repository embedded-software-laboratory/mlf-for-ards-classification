from typing import Union, Dict

from pydantic import  BaseModel

class ImputationMetaData(BaseModel):
    impute_empty_cells: bool
    imputation_parameter_algorithm_dict: Union[Dict[str, str], None]

class UnitConversionMetaData(BaseModel):
    conversions: Dict[str, str]

class ParamCalculationMetaData(BaseModel):
    calculated_parameters: list

class OnsetDetectionMetaData(BaseModel):
    onset_detection_algorithm: str
    onset_detection_return_type: str
    series_begin: Union[float, None]
    series_end: Union[float, None]
    remove_ards_patients_without_onset: bool
    update_ards_diagnose: bool
    fill_cells_out_of_patient_stay: Union[bool, None]

class FilteringMetaData(BaseModel):
    applied_filters: Union[list, None]

class FeatureSelectionMetaData(BaseModel):
    feature_selection_algorithm: str
    min_required_variance: Union[float, None]
    num_features_to_select: Union[int, None]

class TimeseriesMetaData(BaseModel):
    datasource: str
    dataset_location: str

    imputation: Union[ImputationMetaData, None]
    unit_conversion: Union[UnitConversionMetaData, None]
    parameter_calculation: Union[ParamCalculationMetaData, None]
    onset_detection: Union[OnsetDetectionMetaData, None]

    filtering: Union[FilteringMetaData, None]
    feature_selection: Union[FeatureSelectionMetaData, None]
    percentage_of_ards: float
    cross_validation_random_state: Union[int, None]
    cross_validation_shuffle: Union[bool, None]
    cross_validation_folds: Union[int, None]
