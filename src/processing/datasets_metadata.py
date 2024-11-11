from typing import Union


from pydantic import  BaseModel

class ImputationMetaData(BaseModel):
    impute_empty_cells: bool
    imputation_parameter_algorithm_dict: Union[dict[str, str], None]

class UnitConversionMetaData(BaseModel):


class OnsetDetectionMetaData(BaseModel):
    onset_detection_algorithm: str
    onset_detection_return_type: str
    series_begin: Union[float, None]
    series_end: Union[float, None]
    remove_ards_patients_without_onset: bool
    update_ards_diagnose: bool
    fill_cells_out_of_patient_stay: bool



class FeatureSelectionMetaData(BaseModel):
    feature_selection_algorithm: str
    min_required_variance: Union[float, None]
    num_features_to_select: Union[int, None]

class TimeseriesMetaData(BaseModel):
    datasource: str
    dataset_location: str

    imputation: Union[ImputationMetaData, str]
    onset_detection: Union[OnsetDetectionMetaData, str]
    feature_selection: Union[FeatureSelectionMetaData, str]
    applied_filters: Union[list, str]
    percentage_of_ards: float
    cross_validation_random_state: Union[int, None]
