import bisect
import copy
import logging
import math
from multiprocessing import Pool
from typing import Optional, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance
import json

from processing.ad_algorithms.BaseAnomalyDetector import BaseAnomalyDetector
from processing.ad_algorithms.PhysicalLimitsDetector import PhysicalLimitsDetector
from processing.ad_algorithms.torch_utils import check_directory
from processing.datasets_metadata import AnomalyDetectionMetaData

logger = logging.getLogger(__name__)


class SW_ABSAD_Mod_Detector(BaseAnomalyDetector):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = str(kwargs.get('name', "SW_ABSAD_Mod_Detector"))
        self.type = "SW_ABSAD_Mod"
        self.model = None
        self.anomaly_data_dir = str(
            kwargs.get("anomaly_data_dir", f"/work/rwth1474/Data/AnomalyDetection/anomaly_data/SW_ABSAD_Mod/{self.name}"))
        self.prepared_data_dir = str(
            kwargs.get("prepared_data_dir", "/work/rwth1474/Data/AnomalyDetection/windowed_data"))
        check_directory(self.anomaly_data_dir)
        check_directory(self.prepared_data_dir)

        self.replace_zeros = bool(kwargs.get('replace_zeros', False))
        self.replace_physical_outliers = bool(kwargs.get('replace_physical_outliers', False))
        self.use_cl_modification = bool(kwargs.get('use_cl_modification', True))
        self.retrain_after_gap = bool(kwargs.get('retrain_after_gap', True))
        self.window_length = int(kwargs.get('window_length', 100))
        self.variance_check = bool(kwargs.get('variance_check', False))
        self.clean_training = bool(kwargs.get('clean_training_window', True))
        self.variance_window_length = 50
        self.columns_to_check = kwargs["use_columns"].split(",")
        self.bandwidth = 0.5
        self.confidence_level = float(kwargs.get('confidence_level', 0.90))
        self.theta =  float(kwargs.get('theta', 0.5))
        self.k = int(kwargs.get('k', 20))
        self.s = int(kwargs.get('s', 20))
        self.needs_full_data = False
        self.patient_level_anomaly_detection = True

    def create_meta_data(self):
        meta_data_dict = super().create_meta_data()
        meta_data_dict["anomaly_detection_algorithm"] = self.type
        meta_data_dict["algorithm_specific_settings"] = {"replace_zeros": self.replace_zeros,
                                                         "replace_physical_outliers": self.replace_physical_outliers,
                                                         "use_cl_modification": self.use_cl_modification,
                                                         "retrain_after_gap": self.retrain_after_gap,
                                                         "training_window_length": self.window_length,
                                                         "variance_check": self.variance_check,
                                                         "clean_training": self.clean_training,
                                                         "bandwidth": self.bandwidth,
                                                         "confidence_level": self.confidence_level,
                                                         "theta": self.theta,
                                                         "n_nearest_neighbors": self.k,
                                                         "n_reference_windows": self.s,
                                                         }
        if self.variance_check:
            meta_data_dict["algorithm_specific_settings"]["variance_window_length"] = self.variance_window_length
        return AnomalyDetectionMetaData(**meta_data_dict)

    def run(self, dataframe_detection: pd.DataFrame, job_count: int, total_jobs: int) -> (pd.DataFrame, dict[str, dict[str, int]]):
        logger.info(f"Running job {job_count} of {total_jobs} for {self.name}")
        dataframe_detection = self._prepare_data(dataframe_detection)["dataframe"]
        anomaly_dict = self._predict(dataframe_detection)


        fixed_df = pd.concat(anomaly_dict["fixed_dfs"]).reset_index(drop=True)
        anomaly_df = pd.concat(anomaly_dict["anomaly_dfs"]).reset_index(drop=True)
        self._save_anomaly_df(anomaly_df)
        logger.info("Saved data")
        return fixed_df, anomaly_dict["anomaly_count"]


    def _prepare_data(self, dataframe_detection: pd.DataFrame, save_data: bool =False, overwrite: bool = True) -> dict[str, pd.DataFrame]:
        if self.columns_to_check[0]!= '':
            dataframe_detection = dataframe_detection[self.columns_to_check + ["time", "patient_id"]]
        else:
            df_columns = dataframe_detection.columns.tolist()
            df_columns.remove("time")
            df_columns.remove("patient_id")
            self.columns_to_check = df_columns

        if self.replace_zeros:
            dataframe_detection = dataframe_detection.replace(0, np.nan)

        if self.replace_physical_outliers:
            # TODO handle metadata from PhysicalLimitsDetector
            physicalADDetector = PhysicalLimitsDetector(name="PhysicalLimitsDetectorSWABSABMOD", columns_to_check=self.columns_to_check, database=self.database, handling_strategy="delete_value")
            dataframe_detection = physicalADDetector.run(dataframe_detection=dataframe_detection, job_count=-1, total_jobs=-1)

        if self.replace_zeros:
            dataframe_detection = dataframe_detection.replace(0, np.nan)

        return_dict = {"test": dataframe_detection}
        if save_data:
            first_patient, last_patient = self._get_first_and_last_patient_id_for_name(dataframe_detection)
            save_path = f"{self.prepared_data_dir}/patient_{first_patient}_to_{last_patient}.pkl"
            self._save_file(return_dict, save_path, overwrite)
        return return_dict




    def _predict(self, dataframe: pd.DataFrame, **kwargs) -> dict:

        patient_list = dataframe["patient_id"].unique().tolist()
        anomaly_dfs = []
        anomaly_count_dict = {}

        logger.info(patient_list)
        for patient_id in patient_list:
            logger.info(f"Running job for patient {patient_id}...")
            patient_df = dataframe[dataframe["patient_id"] == patient_id]
            if self.columns_to_check[0]!= '':
                relevant_data = patient_df[self.columns_to_check + ["time", "patient_id"]]
            else:
                relevant_data = patient_df
            result_df = self._predict_patient(relevant_data)

            if not result_df.empty:
                anomaly_count_dict = self._calculate_anomaly_count(result_df, patient_df, anomaly_count_dict)
                anomaly_dfs.append(result_df)

            else:
                logger.info(f"Patient {patient_id} has not enough data for prediction.")
                continue

        anomaly_df = pd.concat(anomaly_dfs).reset_index(drop=True)
        self._save_anomaly_df(anomaly_df)
        logger.info(f"Finished for patients: {patient_list}")
        return {
                "anomaly_count": anomaly_count_dict,
                "anomaly_df": anomaly_df}



    def _predict_patient(self, patient_df: pd.DataFrame) -> Optional[pd.DataFrame]:

        timestamp_column = "time"
        df_with_nan = patient_df.copy(deep=True)
        df = patient_df.dropna(axis=0, how="all")
        df_without_offset = df.drop(columns=[timestamp_column, "patient_id"])
        previous_timestamp = 0
        interval = []
        for timestamp in df[timestamp_column].to_numpy():
            interval.append(timestamp - previous_timestamp)
            previous_timestamp = timestamp
        sorted_interval = sorted(((interval.count(e), e) for e in set(interval)), reverse=True)
        count, default_time_interval = sorted_interval[0]
        if self.window_length >= len(df):
            return pd.DataFrame()


        # Dimensionen jedes Datenpunktes in samples_normalized
        self.num_data_dimensions = len(df_without_offset.columns)
        # Tabelle, die die Varianz für jeden Punkt in jeder Dimension speichert
        variance_table = np.array(
            [[np.nan for _ in range(len(df_without_offset))] for _ in range(self.num_data_dimensions)])
        # Tabelle, die nur Nuller-Varianzen speichert. Die andere Tabelle wird auch zum plotten benutzt.
        variance_zeroes = np.array([np.nan for _ in range(len(df_without_offset))])
        # Übersicht über das LOS aller Punkte
        LOS_complete = np.array([0.0 for _ in range(len(df_without_offset) + 10)])
        # Übersicht über die outlier Dimensonen aller Punkte und nicht nur der im aktuellen sliding window
        LOSsubsortges = np.array(
            [[np.nan for _ in range(self.num_data_dimensions)] for _ in range(len(df_without_offset) + 10)])
        # Control limt, anhand dessen entschieden wird, ob ein Datenpunkt ein zu hohes LOS hat.
        cl_complete_table = np.array([0.0 for _ in range(len(df_without_offset))])
        # Tabelle die für jeden Punkt speichert, ob er ein outlier ist.
        outlier_table = np.array([np.nan for _ in range(len(df_without_offset.index) + 1)])
        # Tabelle, der Cosinus Distanz zwischen jedem Punktepaar
        distance_table = np.array(
            [[np.nan for _ in range(self.window_length)] for _ in range(self.window_length)])
        # Die Punkte, die im ersten betrachteten sliding window sind, nicht normalisiert
        sliding_window_unnormalized = df_without_offset.copy().head(self.window_length).to_numpy()
        # Speichert die Varianz für das aktuelle Sliding Window
        sliding_window_variance = df_without_offset.copy().head(self.variance_window_length).to_numpy()
        mad_table = np.array([np.nan for _ in range(self.num_data_dimensions)])
        '''
        Alle Parameter werden in eine samples-Matrix eingefügt und z-normalisiert. Dies hat zur
        Folge, dass alle Parameter gleich gewichtet werden. Besonders wenn sich die Werte,
        wie etwa Körperkerntemperatur und Herzfrequenz, stark in der Höhe ihrer Messwerte
        und den absoluten Differenzen der Schwankungen unterscheiden, ist dies wichtig um
        Verzerrungen zu verhindern.
        '''
        sliding_window_normalized = copy.deepcopy(sliding_window_unnormalized)
        for column_number in range(np.size(sliding_window_normalized, 1)):
            mad_table[column_number] = stats.median_abs_deviation(sliding_window_unnormalized[:, column_number],
                                                                  nan_policy='omit')
            zscore_column = stats.zscore(sliding_window_normalized[:, column_number],
                                         nan_policy='omit')  # was: propagate
            zscore_column = np.nan_to_num(zscore_column)
            sliding_window_normalized[:, column_number] = zscore_column
        # Tabelle mit den Datenpunkten. Nicht relevante subspaces werden durch nan-Werte ersetzt
        relevant_subspaces = np.array(
            [[0.0 for _ in range(self.num_data_dimensions)] for _ in range(self.window_length)])
        # gibt für jeden Punkt(Zeile) an, ob ein anderer Punkt(Index der Spalte) ein K-nearest neighbor
        kNN = np.array([[0 for _ in range(self.window_length)] for _ in range(self.window_length)])
        # Tabelle, die für jeden Punkt(Zeile) nach similarity absteigend geordnet den Index des Punktes
        # und des zugehörigen values angibt
        kNNDist = np.array([[(0.0, 0.0) for _ in range(self.window_length)] for _ in range(self.window_length)])
        # Tabelle gibt für jedes Knotenpaar die Anzahl an K-nearest neighbors an.
        sNN = np.array([[np.nan for _ in range(self.window_length)] for _ in range(self.window_length)])
        LOS_window = np.array([0.0 for _ in range(self.window_length)])
        # Confidenz Intervall Grenze für die Ermittlung des Control Limit
        diag_matrix = np.zeros((self.num_data_dimensions, self.num_data_dimensions), int)
        np.fill_diagonal(diag_matrix, 1)
        training_required = True
        training_windows = np.array([0 for _ in range(len(df_without_offset))])
        sample_counter = 0
        number_outlier_total = 0
        number_outlier_series = 0
        number_outlier_treshold = 5
        next_position = 0
        next_variance_position = 0
        #print(sliding_window_normalized.shape)
        while sample_counter < len(df_without_offset):


            '''
            Konsistenzcheck. Prüft ob eine große Lücke zwischen den letzten Messungen war.
            Es müssen noch mindestestens 2*window-length Punkte übrig sein
            Ansonsten wird das Training neugestartet.
            '''
            if sample_counter < (len(df_without_offset) - 2 * self.window_length):
                f1 = (df[timestamp_column].iloc[sample_counter - 1])
                f2 = (df[timestamp_column].iloc[sample_counter])
                if (f2 - f1) > (default_time_interval * 10):  # TODO change
                    if self.retrain_after_gap:
                        #print("Gap detected: Retraining")
                        training_required = True
            if training_required:
                cleaned_rows = []
                if (self.clean_training):
                    mahalanobis_distances = []
                    current_row = sample_counter
                    for single_row in sliding_window_normalized:
                        mean_vector = np.mean(sliding_window_normalized, axis=0)
                        V = np.cov(sliding_window_normalized.T)

                        p = np.linalg.pinv(V)
                        D = distance.mahalanobis(single_row, mean_vector, p)
                        mahalanobis_distances.append((current_row, D))
                        current_row = current_row + 1
                    mahalanobis_distances.sort(key=lambda x: x[1], reverse=True)
                    clean_training_threshold = int(self.window_length * 0.1)  # 10%
                    for i in range(clean_training_threshold):
                        (row, _) = mahalanobis_distances[i]
                        if (row not in cleaned_rows):
                            cleaned_rows.append(row)
                            df_without_offset.iloc[row] = np.array(
                                [np.nan for _ in range(self.num_data_dimensions)])
                training_sample_counter = 0
                # Tabelle, der Cosinus Distanz zwischen jedem Punktepaar
                distance_table = np.array(
                    [[np.nan for _ in range(self.window_length)] for _ in range(self.window_length)])
                # Die Punkte, die im ersten betrachteten sliding window sind, nicht normalisiert
                sliding_window_unnormalized = df_without_offset.copy()[
                                              sample_counter:sample_counter + self.window_length].to_numpy()
                # Speichert die Varianz für das aktuelle Sliding Window
                sliding_window_variance = df_without_offset.copy()[
                                          sample_counter:sample_counter + self.variance_window_length].to_numpy()
                '''
                Alle Parameter werden in eine samples-Matrix eingefügt und z-normalisiert. Dies hat zur
                Folge, dass alle Parameter gleich gewichtet werden. Besonders wenn sich die Werte,
                wie etwa Körperkerntemperatur und Herzfrequenz, stark in der Höhe ihrer Messwerte
                und den absoluten Differenzen der Schwankungen unterscheiden, ist dies wichtig um
                Verzerrungen zu verhindern.
                '''
                sliding_window_normalized = copy.deepcopy(sliding_window_unnormalized)
                for column_number in range(np.size(sliding_window_normalized, 1)):
                    zscore_column = stats.zscore(sliding_window_normalized[:, column_number],
                                                 nan_policy='omit')
                    zscore_column = np.nan_to_num(zscore_column)
                    sliding_window_normalized[:, column_number] = zscore_column
                # Tabelle mit den Datenpunkten. Nicht relevante subspaces werden durch nan-Werte ersetzt
                relevant_subspaces = np.array(
                    [[0.0 for _ in range(self.num_data_dimensions)] for _ in range(self.window_length)])
                # gibt für jeden Punkt(Zeile) an, ob ein anderer Punkt(Index der Spalte) ein K-nearest neighbor
                kNN = np.array([[0 for _ in range(self.window_length)] for _ in range(self.window_length)])
                # Tabelle, die für jeden Punkt(Zeile) nach similarity absteigend geordnet den Index des Punktes
                # und des zugehörigen values angibt
                kNNDist = np.array(
                    [[(0.0, 0.0) for _ in range(self.window_length)] for _ in range(self.window_length)])
                # Tabelle gibt für jedes Knotenpaar die Anzahl an K-nearest neighbors an.
                sNN = np.array([[np.nan for _ in range(self.window_length)] for _ in range(self.window_length)])
                LOS_window = np.array([0.0 for _ in range(self.window_length)])
                # Confidenz Intervall Grenze für die Ermittlung des Control Limit
                diag_matrix = np.zeros((self.num_data_dimensions, self.num_data_dimensions), int)
                np.fill_diagonal(diag_matrix, 1)
                next_variance_position = 0
                '''
                Wir Berechnen die Varianz für jede Dimension im Sliding Window
                '''
                for column_number in range(self.num_data_dimensions):
                    for current_point in range(self.window_length):
                        variance_table[column_number][current_point] = np.nanvar(
                            sliding_window_variance[:, column_number])
                        next_variance_position = (next_variance_position + 1) % self.variance_window_length
                '''
                Die Bestimmung des Referenzdatensatzes jedes Punktes erfolgt mithilfe eines Shared
                Nearest Neighbor (SNN) Algorithmus. Der Referenzdatensatz soll für einen Punkt
                die Datenpunkte enthalten, die ebenfalls im ersten Sliding Window liegen und ihm
                besonders ähnlich sind. Dazu wird zunächst die Distanz zwischen allen Punkten
                paarweise berechnet und in einer Matrix gespeichert.
                '''
                for current_point in range(self.window_length):
                    for j in range(self.window_length):
                        distance_table[current_point, j] = self.get_cosine_similarity(
                            sliding_window_normalized[current_point], sliding_window_normalized[j])
                '''
                Auf Basis dieser einfachen Distanzberechnung werden für
                jeden Datenpunkt seine k nächsten Nachbarn bestimmt. Dazu werden Punktindizes
                nach abnehmender Distanz zu dem Datenpunkt geordnet und die k ersten Indizes
                dieser Liste gewählt. Um möglichst einfach auf diese Werte zugreifen zu können
                und später auch einfach Änderungen für einzelne Punkte vornehmen zu können,
                wird in einer KNN-Matrix für jeden Punkt die Beziehung zu jedem anderen Punkt
                mit 1, für k nächster Nachbar, oder 0 kodiert. Zudem wird für jeden Datenpunkt
                eine Rangfolge seiner Nachbarn mit der zugehörigen Distanz als KNNDist-Matrix
                gespeichert.
                '''
                for current_point in range(self.window_length):
                    for j in range(self.window_length):
                        kNNDist[current_point, j] = (
                            np.argsort(distance_table[current_point])[self.window_length - (j + 1)],
                            np.sort(distance_table[current_point])[self.window_length - (j + 1)])
                    for k in np.argsort(distance_table[current_point])[-self.k:]:
                        kNN[current_point, k] = 1
                '''
                Mit Hilfe der Kodierung der k nächsten Nachbarn kann die Nummer
                der geteilten Nachbarn für jedes Punktepaar als Summe über die Verundung der
                jeweiligen Zeilen in der KNN-Matrix bestimmt und in einer SNN-Matrix gespeichert
                werden. Eine Shared Nearest Neighbor Nummer kann definiert werden, die angibt
                wie viele gemeinsame k nächsten Nachbarn zwei Punkte haben müssen, um zum
                Referenzdatensatz des jeweils anderen Datenpunkts gerechnet zu werden.
                '''
                for current_point in range(self.window_length):
                    for j in range(self.window_length):
                        is_in_knn_bool = np.logical_and(kNN[current_point], kNN[j])
                        sNN[current_point, j] = sum([int(e) for e in is_in_knn_bool])
                '''
                Im vierten Arbeitsschritt werden die relevanten Subspaces jedes Datenpunkts bestimmt,
                alle anderen müssen nicht bei der Ausreißererkennung betrachtet werden.
                Eine relevante Subspace ist eine Dimension, die sich stark von den anderen Werten
                dieser Dimension der Punkte im Referenzdatensatz, unterscheidet. Zur Bestimmung
                der relevanten Dimensionen wird zunächst der Vektor als Differenz zwischen dem
                Datenvektor des aktuell betrachteten Datenpunkts und dem Mittelwert des Referenzdatensatzes
                berechnet. Zur Bestimmung von relevanten Dimensionen wird der
                Cosinus zwischen dem Vektor und dem Vektor der zugehörigen Achse betrachtet.
                Ist dieser Wert nahe 0, kann angenommen werden, dass in dieser Dimension der
                Unterschied zwischen dem Punkt und seinem Referenzdatensatz gering ist und die
                Dimension wahrscheinlich keine Abweichung zu den anderen Punkten aufweist. Ist
                der Wert hingegen dem Wert 1 nahe, liegt eine signifikante Abweichung vor und
                die Dimension sollte auf jeden Fall weiterhin berücksichtigt werden.
                '''
                for current_point in range(self.window_length):
                    relevant_points = self.get_reference_set(snn=sNN[current_point], data=sliding_window_normalized)
                    mean_vector = relevant_points.mean(axis=0)
                    current_vector = sliding_window_normalized[current_point]
                    subspace = self.calculate_relevant_subspaces(current_vector, mean_vector)
                    relevant_subspace_col = np.vstack([sliding_window_normalized[current_point], subspace])
                    relevant_subspace_col[0, :][relevant_subspace_col[1, :] == 0] = np.nan
                    relevant_subspace_col = np.delete(relevant_subspace_col, 1, 0)
                    relevant_subspace_col = np.squeeze(relevant_subspace_col, axis=0)
                    relevant_subspaces[current_point] = relevant_subspace_col
                matrix_subspace = np.copy(relevant_subspaces)
                matrix_subspace[~np.isnan(matrix_subspace)] = 1
                matrix_subspace = np.nan_to_num(matrix_subspace)
                while training_sample_counter < self.window_length:
                    if (sample_counter + training_sample_counter) in cleaned_rows:
                        training_windows[sample_counter + training_sample_counter] = -1
                    else:
                        training_windows[sample_counter + training_sample_counter] = 1
                    relevant_points = self.get_reference_set(snn=sNN[training_sample_counter],
                                                             data=sliding_window_normalized)
                    '''
                    Die Berechnung des Local Outlier Scores (LOS) erfolgt durch die Bestimmung
                    der normalisierten Mahalanobisdistanz zu den Punkten im Referenzdatensatz, unter
                    Berücksichtigung der relevanten Subspaces.
                    '''
                    if np.all(matrix_subspace[training_sample_counter] == 0):
                        '''
                        Falls keine Dimension des Datenpunktes
                        als relevant betrachtet wird, kann der LOS direkt auf 0 gesetzt werden, da
                        keine Dimension von den Werten des Referenzdatensatzes abweicht und somit der
                        Datenpunkt auch kein Ausreißer sein kann.
                        '''
                        LOS_complete[sample_counter + training_sample_counter] = 0  # all subspaces irrelevant!
                        LOS_window[training_sample_counter] = 0
                    else:
                        '''
                        Der LOS berechnet sich als normierte Mahalanobisdistanz zwischen den
                        relevanten Dimensionen des Datenpunkts, die nicht NaN-Werte der Matrix relevantsub,
                        und diesen Dimensionen des Mittelwertvektors seines Referenzdatensatzes.
                        '''
                        ds = self.calculate_local_outlier_score(points=relevant_points,
                                                                subspaces=relevant_subspaces[
                                                                    training_sample_counter])
                        LOS_complete[sample_counter + training_sample_counter] = ds
                        LOS_window[training_sample_counter] = ds
                    # LOS für jede Subspace
                    mean_vector = relevant_points.mean(axis=0)
                    point_vector = sliding_window_normalized[training_sample_counter]
                    # Sortierung welche Dimension am entferntesten ist
                    if np.linalg.norm(point_vector - mean_vector) == 0.0:
                        LOSsubsort_point = LOS_complete[sample_counter + training_sample_counter]
                    else:
                        LOSsubsort_point = LOS_complete[sample_counter + training_sample_counter] * (
                                np.absolute(point_vector - mean_vector) / (
                            np.linalg.norm(point_vector - mean_vector)))
                    LOSsubsortges[sample_counter + training_sample_counter] = np.argsort(LOSsubsort_point)
                    # Iteration step
                    training_sample_counter = training_sample_counter + 1
                '''
                Im letzten Schritt erfolgt die Bestimmung des Control limits (CL), das den Wert des
                LOS angibt, anhand dessen Ausreißer von normalen Punkten unterschieden werden.
                '''
                CL = self.calculate_control_limit(LOS_window)
                if CL == -1:
                    continue
                for current_point in range(self.window_length):
                    cl_complete_table[sample_counter + current_point] = CL
                '''
                End of Training phase
                '''
                training_required = False
                sample_counter = sample_counter + training_sample_counter
                #print("Training phase completed")
            '''
            Im Anschluss an die Trainingsphase wird jeweils ein neuer Punkt zum Sliding
            Window hinzugefügt und als möglicher Ausreißer getestet. Dieser Punkt ersetzt den
            frühesten Wert im jeweiligen Sliding Window in der samples-Matrix.
            '''

            _next_position_save = next_position
            save = self.TablePositionStruct(len(df_without_offset), self.num_data_dimensions, self.window_length)
            save.next_position = next_position
            save.sliding_window_normalized = copy.deepcopy(sliding_window_normalized)
            save.distance_table = copy.deepcopy(distance_table)
            save.sNN = copy.deepcopy(sNN)
            save.kNN = copy.deepcopy(kNN)
            save.kNNDist = copy.deepcopy(kNNDist)
            save.relevant_subspaces = copy.deepcopy(relevant_subspaces)
            save.LOS_window = copy.deepcopy(LOS_window)
            # Adding new point
            sliding_window_unnormalized[next_position] = df_without_offset.iloc[sample_counter]
            current_point = next_position
            next_position = (next_position + 1) % self.window_length
            '''
            Die Varianz-Tabelle wird geupdatet indem der neue Wert dem Sliding Window
            hinzugefügt wird und die Varianz neuberechnet wird
            '''
            sliding_window_variance[next_variance_position] = df_without_offset.iloc[sample_counter]
            for column_number in range(self.num_data_dimensions):
                variance_table[column_number][sample_counter] = np.nanvar(sliding_window_variance[:, column_number])
            next_variance_position = (next_variance_position + 1) % self.variance_window_length
            '''
            Die Normalisierung wurde in der Implementierung so umgesetzt, dass für alle Punkte,
            die im aktuellen Sliding Window liegen, die ursprünglichen unnormalisierten Werte
            gewählt werden und diese spaltenweise z-normalisiert werden.
            '''
            sliding_window_normalized = copy.deepcopy(sliding_window_unnormalized)
            for column_number in range(np.size(sliding_window_normalized, 1)):
                zscore_column = stats.zscore(sliding_window_normalized[:, column_number],
                                             nan_policy='omit')  # was: propagate
                zscore_column = np.nan_to_num(zscore_column)
                sliding_window_normalized[:, column_number] = zscore_column
            '''
            Zur Bestimmung des Referenzdatensatzes muss die Distanztabelle geupdated werden
            '''
            for j in range(self.window_length):
                distance_table[current_point, j] = self.get_cosine_similarity(
                    sliding_window_normalized[current_point],
                    sliding_window_normalized[j])
            '''
            Zur Bestimmung des Referenzdatensatzes muss die KNN Tabelle geupdated werden. Die
            KNN-Matrix kann für jeden Punkt j, abhängig von dem Wert des neu hinzugefügten
            Punktes und des aus dem Sliding Window herausgenommenen Punktes, angepasst
            werden.
            '''
            for window_point in range(self.window_length):
                if kNN[window_point, current_point] == 1:
                    next_neighbor = kNNDist[window_point, self.k]
                    if distance_table[window_point][current_point] <= next_neighbor[1]:
                        kNN[window_point, int(next_neighbor[0])] = 1
                        kNN[window_point, current_point] = 0
                else:
                    k_neighbor = kNNDist[window_point, self.k]
                    if distance_table[window_point, current_point] > k_neighbor[1]:
                        kNN[window_point, current_point] = 1
                        kNN[window_point, int(k_neighbor[0])] = 0
                kNNDist_without_current_point = [i for i in kNNDist[window_point] if i[0] != current_point]
                _keys = [r[1] for r in kNNDist_without_current_point]
                _index = bisect.bisect(_keys, distance_table[window_point, current_point])
                kNNDist_without_current_point.insert(_index,
                                                     (current_point, distance_table[window_point, current_point]))
                kNNDist[window_point] = copy.deepcopy(kNNDist_without_current_point)
            '''
            Zur Bestimmung des Referenzdatensatzes muss die SNN Tabelle geupdated werden
            ODER muss sie? War bisher nicht so! TODO prüfen
            '''
            for j in range(self.window_length):
                is_in_knn_bool = np.logical_and(kNN[current_point], kNN[j])
                sNN[current_point, j] = sum([int(e) for e in is_in_knn_bool])
            relevant_points = self.get_reference_set(snn=sNN[current_point], data=sliding_window_normalized)
            if len(relevant_points):
                mean_vector = relevant_points.mean(axis=0)
                point_vector = sliding_window_normalized[current_point]
                subspace = self.calculate_relevant_subspaces(point_vector, mean_vector)
            else:
                point_vector = sliding_window_normalized[current_point]
                mean_vector = np.array([0 for _ in range(len(point_vector))])
                subspace = np.array([0 for _ in range(len(point_vector))])
            relevant_subspace_col = np.vstack([sliding_window_normalized[current_point], subspace])
            relevant_subspace_col[0, :][relevant_subspace_col[1, :] == 0] = np.nan
            relevant_subspace_col = np.delete(relevant_subspace_col, 1, 0)
            relevant_subspace_col = np.squeeze(relevant_subspace_col, axis=0)
            relevant_subspaces[current_point] = relevant_subspace_col
            matrix_subspace = copy.deepcopy(relevant_subspaces)
            matrix_subspace[~np.isnan(matrix_subspace)] = 1  # new
            matrix_subspace = np.nan_to_num(matrix_subspace)  # replaces Nan with 0
            # if no subspace is relevant
            if np.all(matrix_subspace[current_point] == 0):
                LOS_complete[sample_counter] = 0
                LOS_window[current_point] = 0
            else:
                ds = self.calculate_local_outlier_score(points=relevant_points,
                                                        subspaces=relevant_subspaces[current_point])
                LOS_complete[sample_counter] = ds
                LOS_window[current_point] = ds
            # Sortierung welche Dimension am entferntesten ist
            if np.linalg.norm(point_vector - mean_vector) == 0.0:
                LOSsubsort_point = LOS_complete[sample_counter]
            else:
                LOSsubsort_point = LOS_complete[sample_counter] * (
                        np.absolute(point_vector - mean_vector) / (np.linalg.norm(point_vector - mean_vector)))
            LOSsubsortges[sample_counter] = np.argsort(LOSsubsort_point)
            if self.variance_check:
                for column_number in range(self.num_data_dimensions):
                    if mad_table[column_number] >= 5.0:
                        if np.round(variance_table[column_number][sample_counter], decimals=2) == 0.0:
                            variance_zeroes[sample_counter] = column_number
                if ~np.isnan(variance_zeroes[sample_counter]):
                    for i in range(self.variance_window_length):
                        LOS_complete[sample_counter - i] = CL + 0.01
                        LOS_window[(current_point - i) % self.window_length] = CL + 0.01
                        outlier_table[sample_counter - i] = 1
                        old_outling_dim = LOSsubsortges[sample_counter - i][self.num_data_dimensions - 1]
                        new_outling_dim = variance_zeroes[sample_counter]
                        old_lossubsortges = LOSsubsortges[sample_counter - i] == old_outling_dim
                        new_lossubsortges = LOSsubsortges[sample_counter - i] == new_outling_dim
                        LOSsubsortges[sample_counter - i][old_lossubsortges] = new_outling_dim
                        LOSsubsortges[sample_counter - i][new_lossubsortges] = old_outling_dim
            if LOS_complete[sample_counter] > CL:
                #print('Outlier detected')
                outlier_table[sample_counter] = 1
            else:
                number_outlier_series = 0
                outlier_table[sample_counter] = 0
            '''
            Das CL wird (falls nötig) geupdated
            '''
            if LOS_complete[sample_counter] > 0:

                if self.use_cl_modification:
                    '''
                    Das neue CL wird bei jedem Punkt berechnet
                    Zur Berechnung wird das gesamte Window benutzt, nicht nur die nicht-outlier.
                    '''
                    if sample_counter < self.window_length:

                        CL = self.calculate_control_limit(LOS_complete[0:sample_counter + 1])
                    else:

                        CL = self.calculate_control_limit(
                            LOS_complete[sample_counter - self.window_length:sample_counter + 1])
                    if CL == -1:
                        continue
                else:
                    '''
                    Das neue CL wird berechnet, wenn der aktuelle Punkt innerhalb des CL liegt
                    = wenn der aktuelle Punkt kein Outlier ist
                    '''
                    if LOS_complete[sample_counter] <= CL:
                        CL = self.calculate_control_limit(LOS_window)
                        if CL == -1:
                            continue

            cl_complete_table[sample_counter] = CL
            '''
            Falls der Punkt als Outlier gefunden wurde, werden die Werte wiederhergestellt.
            Der Outlier soll nicht die Werte verfälschen.
            '''
            if outlier_table[sample_counter] == 1:
                next_position = save.next_position
                sliding_window_normalized = save.sliding_window_normalized
                distance_table = save.distance_table
                kNN = save.kNN
                kNNDist = save.kNNDist
                sNN = save.sNN
                relevant_subspaces = save.relevant_subspaces
                LOS_window = save.LOS_window
                number_outlier_total = number_outlier_total + 1
                number_outlier_series = number_outlier_series + 1
                #if number_outlier_series > number_outlier_treshold:
                #    print('Multiple consecutive outlier detected (Treshold: %s)' % str(number_outlier_treshold))
            # Iteration step
            sample_counter = sample_counter + 1
        df['outlier_table'] = outlier_table[:-1]
        df['relevantcol'] = (LOSsubsortges[:, self.num_data_dimensions - 1])[:-10]
        df['LOS'] = LOS_complete[:-10]
        df['controllimit'] = cl_complete_table
        '''
        Jetzt werden die erkannten Outlier zu einem Result-Set hinzugefügt.
        Dieses wird vom Programm benötigt, um die Outlier zu markieren.
        Es wird unterschieden zwischen Class-1 Outlier (Outlier in der Dimension)
        und Class-2 Outlier (Outlier in korrelierender Dimension)
        '''
        result = {}
        for column_number in range(1, len(df.columns)):  # for every column
            column_name = df.columns[column_number]
            result[column_name] = {}
            outlier_for_column = {}
            _counter = 0
            for index, row in df.iterrows():
                outlier_for_column[index] = row['outlier_table']
            # for every time value
            for key, value in outlier_for_column.items():
                if training_windows[_counter] == 1:
                    outlier_for_column[key] = -1  # Trainingdata
                elif training_windows[_counter] == -1:
                    outlier_for_column[key] = -2
                elif np.isnan(outlier_for_column[key]) or outlier_for_column[key] == 0:
                    # not a outlier!
                    outlier_for_column[key] = 0
                elif outlier_for_column[key] == 1 and LOSsubsortges[
                    _counter, self.num_data_dimensions - 1] == column_number - 1:
                    # level 1 outlier!
                    outlier_for_column[key] = 1
                else:
                    # level 2 outlier!
                    outlier_for_column[key] = 2
                _counter = _counter + 1
            for index, row in df_with_nan.iterrows():
                if column_name in df_with_nan.columns:
                    if np.isnan(row[column_name]):
                        outlier_for_column[index] = 0
            result[column_name] = outlier_for_column

        result_df = pd.DataFrame.from_dict(result, orient='index').T
        result_df.drop(columns=['outlier_table', 'relevantcol', 'LOS', 'controllimit'], inplace=True)

        static_colums = self.columns_not_to_check
        static_colums = [x for x in static_colums if x in result_df.columns]
        result_df.drop(columns=static_colums, inplace=True)
        result_df.replace(0, False, inplace=True)
        result_df.replace(1, True, inplace=True)
        result_df.replace(-1, True, inplace=True)
        result_df.replace(-2, False, inplace=True)
        result_df.replace(2, False, inplace=True)
        result_df["time"] = df["time"]
        result_df["patient_id"] = df["patient_id"]



        return result_df

    @staticmethod
    def get_cosine_similarity(x, y):
        np.seterr('ignore')
        return abs(np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y))))

    def get_reference_set(self, snn, data):
        _ds = np.copy(data)
        _ds = np.append(_ds, np.c_[snn], axis=1)
        _ds = _ds[_ds[:, self.num_data_dimensions].argsort()]  # Fixed!
        _ds = _ds[::-1][:self.s]
        _ds = np.delete(_ds, self.num_data_dimensions, 1)
        return _ds

    @staticmethod
    def get_column_subset(datasets, columns):
        _df = datasets.iloc[:, 0].to_frame()
        for column_number in columns:
            single_column = datasets.iloc[:, int(column_number)]
            _df = _df.join(single_column)
        return _df

    def calculate_relevant_subspaces(self, current_vector, mean_vector):
        """
        Dabei wird der jte Wert des Vektors jeweils einzeln mit jeder anderen Dimension,
        die nicht j ist, hier dargestellt durch 𝑙𝑗− verrechnet, um den Durchschnittswert
        über alle n-1 Kombinationen an 2D Räumen zu bekommen, die die Dimension j
        enthalten. Sollte der Datenpunkt in einer Dimension genau mit dem Mittelwert des
        Referenzdatensatzes übereinstimmen, also der Wert des Vektor gleich 0 sein, wird
        dieser Wert durch einen kleinen positiven Wert, zum Beispiel 10−5 ersetzt um eine
        Division durch 0 zu verhindern.

        Liegt der Durchschnittswert für eine Dimension j unter einem Grenzwert, wird
        diese Dimension als unwichtig angesehen und im Folgenden nicht mehr betrachtet.
        Der Grenzwert T wird mithilfe von Formel 4.2, als durchschnittlicher Wert aller
        paarweisen Cosinus Werte aller Dimensionen multipliziert mit 1 + 𝜃, definiert.

        Berechnung (4.1)

        Parameters
        ----------
        current_vector
        mean_vector
        """
        l_vector = current_vector - mean_vector
        l_vector[l_vector == 0] = 10 ** -5

        pcos = [0.0 for _ in range(self.num_data_dimensions)]
        for j in range(self.num_data_dimensions):
            pcosj = 0.0
            for h in range(self.num_data_dimensions):
                if j != h:
                    pcosj = pcosj + (abs(l_vector[j] / math.sqrt(l_vector[j] ** 2 + l_vector[h] ** 2)))

            pcos[j] = pcosj * (1 / (self.num_data_dimensions - 1))

        # Berechnung (4.2)
        g_treshhold = (1 + self.theta) * (1 / self.num_data_dimensions) * sum(pcos)
        pcos_above_treshold = (pcos > g_treshhold)
        return pcos_above_treshold.astype(int)

    def calculate_control_limit(self, los_sliding_window):
        """
        Das CL wird mithilfe eines Kernel Density Estimator (KDE) berechnet. Der KDE
        gibt dabei auf Basis der vorliegenden Stichprobe an LOSs eine Wahrscheinlichkeitsdichteverteilung
        an, die die Wahrscheinlichkeit des Vorliegens eines Punktes mit
        diesem Wert wiedergeben soll. Der Kernel ist hierbei die Gewichtung der Punkte,
        die zur Berechnung einer entsprechenden Kurve verwendet werden. Besonders häufig
        wird der Gaußsche Kernel verwendet.
        """
        min_value = los_sliding_window[los_sliding_window != 0]
        if len(min_value) == 0:
            min_value = [0.00002]

        '''
        Da Werte die genau bei 0 liegen die
        Kurve überproportional beeinflussen könnten, werden vor der Berechnung des CL
        alle 0 Werte der LOS Liste auf die Hälfte des Minimums der nicht Nullwerte gesetzt.
        '''
        for current_point in range(len(los_sliding_window)):
            if los_sliding_window[current_point] == 0:
                los_sliding_window[current_point] = min(min_value) / 2

        kde = stats.gaussian_kde(los_sliding_window[~np.isnan(los_sliding_window)], self.bandwidth)

        '''
        In der konkreten Implementierung wird die
        gaussian_kde.integrate_box_1d Funktion des scipy.stats packages in Python gewählt.
        Dabei wird langsam ein Schwellenwert angenähert, bis das Konfidenzintervall
        einem vorgegebenem Wert entspricht. Als untere Schranke des evaluierten Bereichs wurde -10 gewählt,
        da die LOS-Werte mindestens Wert 0 haben und eine Anpassung der Verteilungskurve über 0 hinaus so
        in die Berechnung des CL mit einbezogen wird.
        '''
        CL = 0
        confidence_interval = 0
        no_change_counter = 0
        while confidence_interval < self.confidence_level:
            old_confidence_interval = confidence_interval
            # first training: and CL < 5
            CL = CL + 0.001
            confidence_interval = kde.integrate_box_1d(-10, CL)
            if not old_confidence_interval != confidence_interval:
                CL = CL + 0.001 * math.pow(1.1, no_change_counter)
                no_change_counter = no_change_counter + 1
                if no_change_counter == 100:
                    logger.info("Failed to calculate CL")
                    return CL
            else:
                CL = CL + 0.001


        return CL

    def calculate_local_outlier_score(self, points, subspaces):
        """
        Zur Berechnung der Mahalanobisdistanz sind der Mittelwertvektor
        und die invertierte Kovarianzmatrix zwischen dem Datenpunkt und
        dem Referenzdatensatz nötig. Der LOS jedes Punktes wird in einer gleichnamigen
        Matrix gespeichert. Das Risiko des Auftretens einer nonsingulären Kovarianz soll
        dabei durch die zuvor beschriebenen Mechanismen zur Beseitigung stark korrelierter
        Werte zwischen Datenpunkt und Referenzdatensatz und durch die entsprechende
        Wahl der Parameter im KNN und SNN-Algorithmus, zur Vermeidung eines leeren
        Referenzdatensatzes, reduziert werden.

        Parameters
        ----------
        points
        subspaces
        """

        mahalanobis_ds = np.vstack([points, subspaces])
        mahalanobis_ds = mahalanobis_ds[:, ~np.any(np.isnan(mahalanobis_ds), axis=0)]

        mahalanobis_vector = mahalanobis_ds[len(mahalanobis_ds) - 1]
        mahalanobis_ds = np.delete(mahalanobis_ds, (len(mahalanobis_ds) - 1), axis=0)

        if len(mahalanobis_vector) == 0:
            return 0
        else:
            mean_vector = np.mean(mahalanobis_ds, axis=0)

            if len(mahalanobis_vector) == 1:
                covariance_matrix = np.cov(mahalanobis_ds, rowvar=False)
                inverted_covariance_matrix = np.linalg.pinv(covariance_matrix.reshape((1, 1)))
            else:
                cov = np.cov(mahalanobis_ds.T)
                inverted_covariance_matrix = np.linalg.pinv(cov)

            md = distance.mahalanobis(mahalanobis_vector, mean_vector, inverted_covariance_matrix)
            return (1 / len(mahalanobis_vector)) * md

    class TablePositionStruct:
        def __init__(self, df_length, dimensions, window_length):
            self.next_position = np.nan

            self.sliding_window_normalized = np.array(
                [[np.nan for _ in range(window_length)] for _ in range(window_length)])

            # Tabelle, der Cosinus Distanz zwischen jedem Punktepaar
            self.distance_table = np.array([[np.nan for _ in range(window_length)] for _ in range(window_length)])

            # Tabelle mit den Datenpunkten. Nicht relevante subspaces werden durch nan-Werte ersetzt
            self.relevant_subspaces = np.array([[0.0 for _ in range(dimensions)] for _ in range(window_length)])

            # gibt für jeden Punkt(Zeile) an, ob ein anderer Punkt(Index der Spalte) ein K-nearest neighbor
            self.kNN = np.array([[0 for _ in range(window_length)] for _ in range(window_length)])

            # Tabelle, die für jeden Punkt(Zeile) nach similarity absteigend geordnet den Index des Punktes
            # und des zugehörigen values angibt
            self.kNNDist = np.array([[(0.0, 0.0) for _ in range(window_length)] for _ in range(window_length)])

            # Tabelle gibt für jedes Knotenpaar die Anzahl an K-nearest neighbors an.
            self.sNN = np.array([[np.nan for _ in range(window_length)] for _ in range(window_length)])

            # Übersicht über das LOS aller Punkte
            self.LOS_complete = np.array([0.0 for _ in range(df_length + 10)])

            self.LOS_window = np.array([0.0 for _ in range(window_length)])
