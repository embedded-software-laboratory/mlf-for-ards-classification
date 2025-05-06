from typing import Union, Dict

from pydantic import  BaseModel

class AnomalyDetectionMetaData(BaseModel):
    name: str
    anomaly_detection_algorithm: str

    anomaly_handling_strategy: str
    anomaly_fixing_algorithm: Union[str, None] = None
    anomaly_threshold: Union[float, None] = None
    columns_checked: Union[list[dict[str, list[str]]], None] = None
    anomaly_statistics: dict[str, Union[int, float]]
    algorithm_specific_settings: dict[str, Union[str, int, float, bool, None]] = None



class ImputationMetaData(BaseModel):
    impute_empty_cells: bool
    imputation_parameter_algorithm_dict: Union[Dict[str, str], None]

class ImputationMetaDataManager:

    @staticmethod
    def merge(old: ImputationMetaData, new: ImputationMetaData) -> ImputationMetaData:

        if old and new:
            impute_empty_cells = old.impute_empty_cells or new.impute_empty_cells
            if not old.imputation_parameter_algorithm_dict and not new.imputation_parameter_algorithm_dict:
                imputation_parameter_algorithm_dict = None
            elif not old.imputation_parameter_algorithm_dict and new.imputation_parameter_algorithm_dict:
                imputation_parameter_algorithm_dict = new.imputation_parameter_algorithm_dict
            elif old.imputation_parameter_algorithm_dict and not new.imputation_parameter_algorithm_dict:
                imputation_parameter_algorithm_dict = old.imputation_parameter_algorithm_dict
            else:
                imputation_parameter_algorithm_dict = old.imputation_parameter_algorithm_dict
                if not "all" in imputation_parameter_algorithm_dict.keys():
                    for key in new.imputation_parameter_algorithm_dict.keys():
                        if key not in imputation_parameter_algorithm_dict.keys():
                            imputation_parameter_algorithm_dict[key] = old.imputation_parameter_algorithm_dict[key]
            return ImputationMetaData(impute_empty_cells=impute_empty_cells, imputation_parameter_algorithm_dict=imputation_parameter_algorithm_dict)
        elif old and not new:
            return old
        else:
            return new




class UnitConversionMetaData(BaseModel):
    conversions: Dict[str, str]

class ParamCalculationMetaData(BaseModel):
    calculated_parameters: list

class ParamCalculationMetaDataManager:

    @staticmethod
    def merge(old: ParamCalculationMetaData, new: ParamCalculationMetaData) -> ParamCalculationMetaData:
        if old and new:
            new_list = old.calculated_parameters + new.calculated_parameters
            return ParamCalculationMetaData(calculated_parameters=new_list)
        elif old:
            return old

        else:
            return new

class OnsetDetectionMetaData(BaseModel):
    onset_detection_algorithm: str
    onset_detection_return_type: str
    series_begin: Union[float, None]
    series_end: Union[float, None]
    remove_ards_patients_without_onset: bool
    update_ards_diagnose: Union[bool, None]
    fill_cells_out_of_patient_stay: Union[bool, None]

class FilteringMetaData(BaseModel):
    applied_filters: Union[list, None]

class FilteringMetaDataManager:

    @staticmethod
    def merge(old: FilteringMetaData, new: FilteringMetaData) -> FilteringMetaData:
        if old and new:
            new_list = old.applied_filters + new.applied_filters
            filter_set = set(new_list)
            if "Lite" in filter_set and "Strict" in filter_set:
                filter_set.remove("Lite")
            filter_list = list(filter_set)
            return FilteringMetaData(applied_filters=filter_list)

        elif old:
            return old

        else:
            return new

class FeatureSelectionMetaData(BaseModel):
    feature_selection_algorithm: str
    min_required_variance: Union[float, None]
    num_features_to_select: Union[int, None]
    first_selection: bool

class FeatureSelectionMetaDataManager:

    @staticmethod
    def merge(old: FeatureSelectionMetaData, new: FeatureSelectionMetaData) -> FeatureSelectionMetaData:
        if old and new:
            first_selection = False
            feature_selection_algorithm = new.feature_selection_algorithm
            min_required_variance = new.min_required_variance
            num_features_to_select = new.num_features_to_select
            return FeatureSelectionMetaData(feature_selection_algorithm=feature_selection_algorithm, min_required_variance=min_required_variance, num_features_to_select=num_features_to_select, first_selection=first_selection)
        elif old:
            return old

        else:
            return new


class TimeseriesMetaData(BaseModel):
    # TODO validate that if is_child is True parent_meta_data_location exists
    # TODO validate that if feature_selection first_selection is False is_child is True

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
    @staticmethod
    def factory_method(processing_meta_data: dict, path: str, percentage_ards: float, dataset_type: str, additional_information: str=None):


        meta_data_dataset = TimeseriesMetaData(datasource=processing_meta_data["database_name"],
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
                                               is_child = False,
                                               parent_meta_data_location=None
                                               )


        return meta_data_dataset

    @staticmethod
    def merge_meta_data(existing_meta_data: TimeseriesMetaData, new_meta_data: TimeseriesMetaData) -> TimeseriesMetaData:
        datasource = existing_meta_data.datasource
        dataset_location = new_meta_data.dataset_location
        dataset_type = new_meta_data.dataset_type
        additional_information = existing_meta_data.additional_information + ", " + new_meta_data.additional_information
        if new_meta_data.imputation or existing_meta_data.imputation:
            imputation = ImputationMetaDataManager.merge(existing_meta_data.imputation, new_meta_data.imputation)
        else:
            imputation = None

        unit_coversion = existing_meta_data.unit_conversion

        if existing_meta_data.parameter_calculation or new_meta_data.parameter_calculation:
            parameter_calculation = ParamCalculationMetaDataManager.merge(existing_meta_data.parameter_calculation, new_meta_data.parameter_calculation)
        else:
            parameter_calculation = None
        onset_detection = existing_meta_data.onset_detection

        if existing_meta_data.filtering or new_meta_data.filtering:
            filtering = FilteringMetaDataManager.merge(existing_meta_data.filtering, new_meta_data.filtering)
        else:
            filtering = None

        if existing_meta_data.feature_selection or new_meta_data.feature_selection:
            feature_selection = FeatureSelectionMetaDataManager.merge(existing_meta_data.feature_selection, new_meta_data.feature_selection)
        else:
            feature_selection = None
        percentage_of_ards = existing_meta_data.percentage_of_ards
        is_child = True
        parent_meta_data_location = existing_meta_data.dataset_location

        return TimeseriesMetaData(datasource=datasource, dataset_location=dataset_location, dataset_type=dataset_type,
                                  additional_information=additional_information, imputation=imputation, unit_conversion=unit_coversion,
                                  parameter_calculation=parameter_calculation, onset_detection=onset_detection, filtering=filtering,
                                  feature_selection=feature_selection, percentage_of_ards=percentage_of_ards, is_child=is_child, parent_meta_data_location=parent_meta_data_location)


    @staticmethod
    def extract_procesing_meta_data(meta_data_dataset: TimeseriesMetaData) -> dict:
        processing_meta_data = {"database_name": meta_data_dataset.datasource,
                                "imputation": meta_data_dataset.imputation,
                                "unit_conversion": meta_data_dataset.unit_conversion,
                                "param_calculation": meta_data_dataset.parameter_calculation,
                                "onset_determination": meta_data_dataset.onset_detection,
                                "filtering": meta_data_dataset.filtering,
                                "feature_selection": meta_data_dataset.feature_selection}
        return processing_meta_data


    @staticmethod
    def write(meta_data_dataset: TimeseriesMetaData):
        path = meta_data_dataset.dataset_location + "_meta_data.json"
        with open(path, "w") as f:
            f.write(meta_data_dataset.model_dump_json(indent=4))

    @staticmethod
    def load_from_dict(meta_data_dict: dict) -> TimeseriesMetaData:
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
        return TimeseriesMetaData(**content)


