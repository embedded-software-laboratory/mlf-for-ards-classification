import statistics

import pandas as pd
import numpy as np


@staticmethod
def _load_numpy_file(file_path) -> pd.DataFrame:
    data = np.load(file_path, mmap_mode="r+")
    variables_file = open(file_path + ".vars")
    variables = variables_file.read().split(" ")
    dataframe = pd.DataFrame(data, columns=variables)
    return dataframe

df = _load_numpy_file("/work/rwth1474/Data/time_series/uka_050623.npy")
time_in_between = {column: [] for column in df.columns if column not in ["patient_id", "time"]}
missing_info = df.isnull().sum()
total_entries = df.shape[0]
missing_entries = []
rel_missing_entries = []
columns = []
for column in df.columns:
    if column not in ["patient_id", "time"]:
        columns.append(column)
        missing = df[column].isnull().sum()
        rel_missing = missing / total_entries
        rel_missing_entries.append(rel_missing)
        missing_entries.append(missing)
missing_info = pd.DataFrame({"columns": columns, "absolute": missing_entries, "relative": rel_missing_entries})
missing_info.to_csv("./missing_values.csv", index=False)
patient_ids = df["patient_id"].unique().tolist()
for patient_id in patient_ids:
    relevant_patient = df[df["patient_id"] == patient_id]

    for column in relevant_patient.columns:
        if column in ["time", "patient_id"]:
            continue
        relevant_data = df[["time", column]]
        relevant_data = relevant_data.dropna(subset=[column]).reset_index(drop=True)
        time_in_between[column] += relevant_data["time"].diff().abs().tolist()
avg_time_in_between = {column: [] for column in df.columns if column not in ["time", "patient_id"]}
for column in time_in_between.keys():
    avg_time_in_between[column] = statistics.mean(time_in_between[column])
total_times = len(df.index) * 15
value_names = []
expected_mising_values = []
true_missing_values = missing_entries
expected_rel_mising_values = []
true_rel_missing_values = []
for column in avg_time_in_between.keys():
    value_names.append(column)
    expected_mising_values.append(total_times/avg_time_in_between[column])

df_overview_present_expected = pd.DataFrame()
df_overview_present_expected["value_name"] = value_names
df_overview_present_expected["num_expected_missing_values"] = expected_mising_values
df_overview_present_expected["num_true_missing_values"] = true_missing_values
df_overview_present_expected.to_csv("./overview_present_expected.csv", index=False)



