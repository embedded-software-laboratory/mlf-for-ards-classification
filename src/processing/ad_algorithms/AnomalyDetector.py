import pandas as pd


class AnomalyDetector:

    def __init__(self, **kwargs):
        self.name = None
        self.type = None
        self.model = None
        self.columns_to_check = None
        self.database = None
        self.fix_algorithm = None
        self.handling_strategy = None
        self.anomaly_threshold = None
        self.max_processes = 1
        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)
        pass

    def run(self, dataframe_training: pd.DataFrame, dataframe_detection: pd.DataFrame, job_count: int, total_jobs: int) -> pd.DataFrame:
        raise NotImplementedError()

    def _train_ad_model(self):
        raise NotImplementedError()

    def _predict(self, dataframe: pd.DataFrame) -> dict:
        raise NotImplementedError()

    def _predict_proba(self):
        raise NotImplementedError()

    def _prepare_data(self, dataframe: pd.DataFrame) -> dict:
        raise NotImplementedError()

    def _handle_anomalies(self, anomalies: dict, anomalous_data : pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def _delete_value(self, anomaly_df: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.mask(anomaly_df)
        return dataframe

    def _delete_than_impute(self, anomaly_df: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.fillna(-100000)
        dataframe = dataframe.mask(anomaly_df)
        dataframe = self._fix_deleted(dataframe)
        dataframe = dataframe.replace(-100000, pd.NA)
        return dataframe

    def _delete_row_if_any_anomaly(self, anomaly_df: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:

        dataframe = dataframe.mask(anomaly_df)
        index_to_drop = []
        for index, row in dataframe.iterrows():
            if row.isnull().any():
                index_to_drop.append(index)
        dataframe = dataframe.drop(index_to_drop)
        return dataframe

    def _delete_row_if_many_anomalies(self, anomaly_df: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:

        n_columns = len(dataframe.columns)
        n_anomalies = anomaly_df.sum(axis=1)
        dataframe = dataframe.fillna(-100000)
        index_to_drop = []
        for i in range(len(dataframe)):
            if n_anomalies[i]/n_columns > self.anomaly_threshold:
                index_to_drop.append(i)
        dataframe = dataframe.drop(index_to_drop)
        dataframe = self._fix_deleted(dataframe)
        dataframe = dataframe.replace(-100000, pd.NA)
        return dataframe

    @staticmethod
    def _add_anomaly_score(anomaly_df: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
        for column in dataframe.columns:
            if column in anomaly_df.columns:
                dataframe[column + "_anomaly"] = anomaly_df[column]
        return dataframe

    def _fix_deleted(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if self.fix_algorithm == "forward":
            dataframe = dataframe.fillna(method="ffill")
        elif self.fix_algorithm == "backward":
            dataframe = dataframe.fillna(method="bfill")
        elif self.fix_algorithm == "interpolate":
            dataframe = dataframe.interpolate(method="linear", limit_direction="both")

        else:
            raise ValueError("Invalid fix_algorithm for Anomaly detection")
        return dataframe

    def _use_prediction(self, prediction: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()



