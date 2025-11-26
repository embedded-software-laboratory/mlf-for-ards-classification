import os

import numpy as np
import pandas as pd

def dict_to_list(ad_dicts):
    all_columns = set()
    for approach in ad_dicts.keys():
        contained_columns = approach.keys()
        for contained_column in contained_columns:
            all_columns.add(contained_column)
    all_columns = list(all_columns)
    values = []
    value_dict = {}
    for approach in ad_dicts.keys():
        contained_info = ad_dicts[approach]
        for contained_column in all_columns:
            if contained_column in contained_info:
                values.append(contained_info[contained_column])
            else:
                values.append(np.nan)
        value_dict[approach] = values
    df = pd.DataFrame.from_dict(value_dict, orient='index', columns=all_columns)
    return df




full_data_file = "/work/rwth1474/Data/time_series/uka_data_050623_testing_all_approaches.pkl"

# TODO read full_features correctly
full_data_df = pd.read_pickle(full_data_file)
contained_patient_ids = full_data_df["patient_id"].unique().tolist()


full_data_cleaned = full_data_df.copy()
count_dict_full_data_cleaned = {}
for column in full_data_cleaned.columns:
    count_dict_full_data_cleaned[column] = full_data_cleaned[column].count()
ad_approaches = ["DeepAnt", "ALAD", "Physical_Outliers", "SW_ABSAD_MOD"]
columns = []
ad_dict_absolute = {}
ad_dict_relative = {}
for ad_approach in ad_approaches:
    anomaly_data_directory = f"/work/rwth1474/Data/AnomalyDetection/anomaly_data/{ad_approach}/"

    ad_dfs = []
    for file in os.listdir(anomaly_data_directory):
        if file.endswith(".pkl"):
            ad_dfs.append(pd.read_pickle(os.path.join(anomaly_data_directory, file)))
    ad_df = pd.concat(ad_dfs)
    ad_df = ad_df[ad_df["patient_id"].isin(contained_patient_ids)]
    ad_df = ad_df.reset_index(drop=True)

    absolute_anomalies = {}
    relative_anomalies = {}

    for column in ad_df.columns:
        if ad_df[column].dtype == "bool":
            if column not in columns:
                columns.append(column)
            anomaly_count = ad_df[column].sum()
            relative_anomaly_count = anomaly_count / count_dict_full_data_cleaned[column]
            absolute_anomalies[column] = anomaly_count
            relative_anomalies[column] = relative_anomaly_count
        else:
            print(f"{column} is not a boolean column")

    ad_dict_relative[ad_approach] = relative_anomalies
    ad_dict_absolute[ad_approach] = absolute_anomalies

relative_ad_df = dict_to_list(ad_dict_relative)
absolute_ad_df = dict_to_list(ad_dict_absolute)
relative_ad_df.to_csv("./RelativeAnomalies.csv")
absolute_ad_df.to_csv("./AbsoluteAnomalies.csv")











