

from processing.datasets_metadata import TimeseriesMetaData, TimeSeriesMetaDataManagement


from pydantic import BaseModel


import pandas as pd







class TimeSeriesDataset(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    content: pd.DataFrame

    meta_data: TimeseriesMetaData

class TimeSeriesDatasetManagement:
    @staticmethod
    def factory_method(dataset: pd.DataFrame, processing_meta_data: dict, path:str, dataset_type: str, additional_information: str=None, existing_meta_data: TimeseriesMetaData=None) -> TimeSeriesDataset:
        if existing_meta_data:
            return TimeSeriesDatasetManagement.factory_method(dataset, processing_meta_data, path, dataset_type, additional_information, existing_meta_data)
        else:
            return TimeSeriesDatasetManagement._factory_new_meta_data(dataset, processing_meta_data, path, dataset_type, additional_information)


    @staticmethod
    def _factory_existing_meta_data(dataset: pd.DataFrame, processing_meta_data: dict, path:str, dataset_type: str, additional_information: str=None, existing_meta_data: TimeseriesMetaData=None):
        new_percentage_ards = dataset["ards"].sum() / len(dataset.index)
        new_meta_data = TimeSeriesMetaDataManagement.factory_method(processing_meta_data, path,
                                                                                       new_percentage_ards, dataset_type,
                                                                                       additional_information)
        return TimeSeriesDataset(content=dataset,
                                 meta_data=TimeSeriesMetaDataManagement.merge_meta_data(existing_meta_data, new_meta_data))
    @staticmethod
    def _factory_new_meta_data(dataset: pd.DataFrame, processing_meta_data: dict, path:str, dataset_type: str, additional_information: str=None):
        percentage_ards = dataset["ards"].sum() / len(dataset.index)
        return TimeSeriesDataset(content=dataset,
                                 meta_data=TimeSeriesMetaDataManagement.factory_method(processing_meta_data, path,
                                                                                       percentage_ards, dataset_type,
                                                                                       additional_information))
    @staticmethod
    def write(timeseries_dataset: TimeSeriesDataset):
        path = timeseries_dataset.meta_data.dataset_location + ".csv"
        timeseries_dataset.content.to_csv(path, index=False, header=True)
        TimeSeriesMetaDataManagement.write(timeseries_dataset.meta_data)
