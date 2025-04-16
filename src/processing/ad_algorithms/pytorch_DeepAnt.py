import math
import os
from typing import Any, Tuple

import numpy as np
import pandas as pd

import logging

import pytorch_lightning as pl
import torch.nn as  nn
import torch

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from processing.ad_algorithms.BaseAnomalyDetector import BaseAnomalyDetector
from processing.ad_algorithms.torchWindowGenerator import WindowGenerator, DataModule
from processing.ad_algorithms.torch_utils import check_device, check_directory, set_seed

logger = logging.getLogger(__name__)

class DeepAntPredictor(nn.Module):

    def __init__(self, feature_dim: int, window_size: int, prediction_size: int, hidden_size: int = 256):
        super(DeepAntPredictor, self).__init__()
        self.feature_dim = feature_dim
        self.window_size = window_size

        self.n_filters = 32 # Taken from paper
        size_after_conv1 = window_size - 2
        size_after_pool1 = (size_after_conv1 - 2 - 1) / 2 + 1
        size_after_conv2 = size_after_pool1 - 2
        size_after_pool2 = (size_after_conv2 - 2 -1 ) / 2 + 1
        self.fc_input_size = math.floor(size_after_pool2) * self.n_filters

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=feature_dim, out_channels=self.n_filters, kernel_size=3, padding="valid"),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=self.n_filters, out_channels=self.n_filters, kernel_size=3, padding="valid"),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=self.fc_input_size, out_features=hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=hidden_size, out_features=prediction_size)
        )

        logger.info("DeepAntPredictor initialized.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass through the DeepAntPredictor.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, feature_dim, window_size).

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, prediction_size).
        """
        return self.model(x)

class AnomalyDetector(pl.LightningModule):

    def __init__(self, model: nn.Module, learning_rate: float = 1e-3):
        """
            Anomaly detector of DeepAnt.

            Args:
                model (nn.Module): The DeepAntPredictor model.
                learning_rate (float): Learning rate for the optimizer.

        """

        super(AnomalyDetector, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.L1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass through the anomaly detector.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, feature_dim, window_size).

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, prediction_size).
        """
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        """
            Defines a single step in the training loop.

            Args:
                batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data.
                    The first element is the input tensor of shape (batch_size, feature_dim, window_size).
                    The second element is the target tensor of shape (batch_size, prediction_size).
                batch_idx (int): Index of the batch.

            Returns:
                torch.Tensor: The training loss value of the current batch.
        """

        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log('train_loss', loss.item(), on_epoch=True)
        logger.info(f"Epoch {self.current_epoch} - Training step {batch_idx} - Loss: {loss.item()}")

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        """
            Defines a single step in the validation loop.

            Args:
                batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data.
                    The first element is the input tensor of shape (batch_size, feature_dim, window_size).
                    The second element is the target tensor of shape (batch_size, prediction_size).
                batch_idx (int): Index of the batch.

            Returns:
                torch.Tensor: The validation loss value of the current batch.
        """

        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.log('val_loss', loss.item(), on_epoch=True)
        logger.info(f"Epoch {self.current_epoch} - Validation step {batch_idx} - Loss: {loss.item()}")
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """
            Defines a step for prediction.

            Args:
                batch (Any): A batch of data.
                    The first element is the input tensor of shape (batch_size, feature_dim, window_size).
                    The second element is the target tensor of shape (batch_size, prediction_size).
                batch_idx (int): Index of the batch.

            Returns:
                torch.Tensor: The predicted values for the current batch.
        """
        x, y = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        """
            Configures the optimizer for the model.

            Returns:
                torch.optim.Optimizer: The optimizer for the model.
        """
        logger.info(f"Configuring optimizer with learning rate: {self.learning_rate}")
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)


class DeepAnt:

    def __init__(self, config: dict, train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset,
                 test_dataset: torch.utils.data.Dataset, feature_dim: int, name: str):

        """
            DeepAnt anomaly detection model.

            Args:
                config (dict): Configuration dictionary containing model parameters.
                train_dataset (torch.utils.data.Dataset): Training dataset.
                val_dataset (torch.utils.data.Dataset): Validation dataset.
                test_dataset (torch.utils.data.Dataset): Test dataset.
                feature_dim (int): Dimension of the input features.
                window_size (int): Size of the input window.
        """

        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.feature_dim = feature_dim
        self.name = name


        self.device = config["device"]
        self.deepant_predictor = DeepAntPredictor(self.feature_dim, config["window_size"], config["prediction_size"],
                                                  config["hidden_size"]).to(self.device)
        self.anomaly_detector = AnomalyDetector(self.deepant_predictor, config["learning_rate"]).to(self.device)
        self.initial_trainer = pl.Trainer(
            max_epochs=self.config["max_initial_epochs"],
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=self.config["checkpoint_dir"],
                    filename=f"initial_model_{self.name}",
                    monitor="epoch",
                    save_top_k=1,
                    mode="max"
                ),
                pl.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.config["patience"],
                    mode = "min"
                )
            ],
            default_root_dir=self.config["run_dir"],
            accelerator=self.device,
            devices=1 if self.device == "cuda" else "auto"
        )

        self.trainer = pl.Trainer(
            max_epochs=self.config["max_epochs"],
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=config["run_dir"],
                    filename=f"best_model_{self.name}",
                    monitor="val_loss",
                    save_top_k=1,
                    mode="min"
                ),
                pl.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.config["patience"],
                    mode = "min"
                )
            ],
            default_root_dir=config["run_dir"],
            accelerator=self.device,
            devices=1 if self.device == "cuda" else "auto"
        )
        logger.info("DeepAnt initialized.")

    def train(self):
        """
            Trains the DeepAnt model using the training dataset in two phases:
            1. Initial short training
            2. Main training, with early stopping based on validation loss.
        """
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.config["batch_size"],
                                  shuffle=True
        )
        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.config["batch_size"],
                                shuffle=False
        )
        logger.info("Starting initial training...")
        self.initial_trainer.fit(self.anomaly_detector, train_loader, val_loader)

        initial_checkpoint_path = os.path.join(self.config["run_dir"], f"initial_model{self.name}.ckpt")
        self.anomaly_detector = AnomalyDetector.load_from_checkpoint(
            checkpoint_path=initial_checkpoint_path,
            model=self.deepant_predictor,
            learning_rate=self.config["learning_rate"]
        )
        logger.info("Initial training completed. Starting main training...")
        self.trainer.fit(self.anomaly_detector, train_loader, val_loader)
        logger.info("Main training completed. Saving the best model...")

    def predict(self) -> dict:
        """
            Detects anomalies in the test dataset using the trained DeepAnt model.

            Returns:
                dict: Dictionary containing indices of detected anomalies for each feature.
        """
        logger.info("Starting detection of anomalies...")
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=self.config["batch_size"],
                                 shuffle=False
        )
        best_model = AnomalyDetector.load_from_checkpoint(checkpoint_path=os.path.join(self.config["run_dir"], f"best_model_{self.name}.ckpt"),
                                                              map_location=self.device,
                                                              lr=self.config["learning_rate"])
        output = self.trainer.predict(best_model, test_loader)

        ground_truth = test_loader.dataset.data_y.squeeze()
        predictions = output[0].numpy().squeeze()

        if ground_truth.ndim == 1:
            ground_truth = ground_truth.reshape(-1, 1)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        anomaly_scores = np.abs(predictions - ground_truth)
        if anomaly_scores.ndim == 1:
            anomaly_scores = anomaly_scores.reshape(-1, 1)
        thresholds = self.calculate_thresholds(anomaly_scores)
        logger.info(f"Calculated thresholds: {thresholds}")



        anomalies_indices = self.identify_anomalies(anomaly_scores, thresholds)
        logger.info(f"Identified anomalies: {anomalies_indices}")
        return anomalies_indices

    def calculate_thresholds(self, anomaly_scores: np.ndarray, std_rate: int =2) -> list[float]:
        """
            Calculates the thresholds for anomaly detection.

            Args:
                anomaly_scores (np.ndarray): Anomaly scores.
                std_rate (int): Number of standard deviation multiplier to use for threshold calculation. Default is 2.

            Returns:
                List[float]: List of thresholds for each feature.
        """



        thresholds = []
        for feature_idx in range(self.feature_dim):
            feature_scores = anomaly_scores[:, feature_idx]
            mean = np.mean(feature_scores)
            std = np.std(feature_scores)
            threshold = mean + std_rate * std
            thresholds.append(threshold)
        return thresholds

    def identify_anomalies(self, anomaly_scores: np.ndarray, thresholds: list[float]) -> dict:

        """
            Identifies anomalies based on the per-feature calculated thresholds.

            Args:
                anomaly_scores (np.ndarray): Anomaly scores.
                thresholds (list[float]): List of thresholds for each feature.

            Returns:
                dict: Dictionary containing indices of detected anomalies for each feature.
        """

        anomaly_dict = {}
        for feature_idx in range(self.feature_dim):
            feature_scores = anomaly_scores[:, feature_idx]
            threshold = thresholds[feature_idx]
            anomalies = np.where(feature_scores > threshold)[0]
            anomaly_dict[f"feature_{feature_idx}"] = anomalies
        return anomaly_dict


class DeepAntDetector(BaseAnomalyDetector):

    def __init__(self, **kwargs):
        """
            DeepAnt anomaly detection model.

            Args:
                kwargs: Configuration dictionary containing model parameters.
        """
        super(DeepAntDetector, self).__init__(**kwargs)
        self.name = str(kwargs.get("name", "DeepAntDetector"))
        self.type = "DeepAnt"
        self.model = None

        self.datasets_to_create = list(kwargs.get('dataset_to_create', []))
        self.val_percentage = float(kwargs.get('val_percentage', 0.1))
        self.train_percentage = float(kwargs.get('train_percentage', 0.1))
        self.test_percentage = float(kwargs.get('test_percentage', 0.1))
        self.sk_seed = int(kwargs.get('seed', 42))
        set_seed(self.sk_seed)
        self.needs_full_data = True
        self.window_generator_config = dict(kwargs.get('window_generator_config', {}))


        self.learning_rate = float(kwargs.get("learning_rate", 1e-3))
        self.hidden_size = int(kwargs.get("hidden_size", 256))
        self.max_epochs = int(kwargs.get("max_epochs", 100))
        self.max_initial_epochs = int(kwargs.get("max_initial_epochs", 10))
        self.batch_size = int(kwargs.get("batch_size", 32))
        self.patience = int(kwargs.get("patience", 5))
        self.run_dir = str(kwargs.get("run_dir", "../Data/Models/AnomalyDetection/DeepAnt"))
        self.checkpoint_dir = str(kwargs.get("checkpoint_dir", "../Data/Models/AnomalyDetection/DeepAnt"))
        check_directory(str(self.run_dir))
        check_directory(str(self.checkpoint_dir))
        self.device = check_device()

        self.deepant_config = {
            "window_size" : int(self.window_generator_config.get("input_width", 20)),
            "prediction_size" : int(self.window_generator_config.get("output_width", 1)),
            "hidden_size" : int(self.hidden_size),
            "learning_rate" : float(self.learning_rate),
            "max_initial_epochs" : int(self.max_initial_epochs),
            "max_epochs" : int(self.max_epochs),
            "checkpoint_dir" : str(self.checkpoint_dir),
            "run_dir" : str(self.run_dir),
            "patience": int(self.patience),
            "device" : str(self.device),
            "batch_size" : int(self.batch_size)
        }

    def run(self,  dataframe_detection: pd.DataFrame, job_count: int, total_jobs: int) -> pd.DataFrame:

        prepared_data = self._prepare_data(dataframe_detection)
        existing_labels = []
        for item in self.datasets_to_create:
            labels = item["labels"]
            for label in labels:
                if label not in existing_labels:
                    existing_labels.append(label)
                else:
                    raise ValueError("Double label detected for label: " + label)

        results = []
        for item in self.datasets_to_create:
            print("Running dataset: ", item["name"])
            results.append(self._run_step(prepared_data, item))

        df_anomaly = pd.DataFrame()
        for item in results:
            if not item.empty:
                df_anomaly.merge(item, how="outer", left_index=True, right_index=True)
        anomaly_columns = df_anomaly.columns.tolist()
        rename_dict = {column: column + "_anomaly" for column in anomaly_columns}
        remaining_data = prepared_data["test"]
        df_anomaly = df_anomaly.rename(columns=rename_dict)
        relevant_data = prepared_data["test"][df_anomaly.columns.tolist()]
        df_anomaly = df_anomaly.fillna(False)
        fixed_df = self._handle_anomalies({"results": df_anomaly}, relevant_data, remaining_data)
        return fixed_df

    @staticmethod
    def _build_patient_dict(dataframe: pd.DataFrame) -> dict:
        patient_dict = {}
        patient_ids = dataframe["patient_id"].unique().tolist()
        patient_max_time_df = dataframe.groupby("patient_id")["time"].idxmax()
        patient_min_time_df = dataframe.groupby("patient_id")["time"].idxmin()
        for i in range(len(patient_ids)):
            patient_dict[patient_ids[i]] = (int(patient_min_time_df.iloc[i]),
                                            int(patient_max_time_df.iloc[i])
                                            )
        return patient_dict

    def _run_step(self, data: dict[str, pd.DataFrame], dataset_to_create: dict) -> pd.DataFrame:
        """
            Runs the anomaly detection step for a specific dataset.

            Args:
                prepared_data (dict): Dictionary containing the prepared data.
                dataset_to_create (dict): Dictionary containing the dataset configuration.

            Returns:
                pd.DataFrame: The processed DataFrame after anomaly detection.
        """
        name = dataset_to_create["name"]
        train = data["train"]
        val = data["val"]
        test = data["test"]

        relevant_columns = dataset_to_create["labels"] + dataset_to_create["features"] + ["patient_id", "time"]
        relevant_train = train[relevant_columns]
        relevant_train = relevant_train.dropna(how='any', axis=0).reset_index(drop=True)
        relevant_val = val[relevant_columns]
        relevant_val = relevant_val.dropna(how='any', axis=0).reset_index(drop=True)
        relevant_test = test[relevant_columns]
        relevant_test = relevant_test.dropna(how='any', axis=0).reset_index(drop=True)
        patient_divisions_train = self._build_patient_dict(relevant_train)
        patient_divisions_val = self._build_patient_dict(relevant_val)
        patient_divisions_test = self._build_patient_dict(relevant_test)

        scaler = MinMaxScaler()
        scaler.fit(relevant_train)
        scaled_train = scaler.transform(relevant_train)
        scaled_val = scaler.transform(relevant_val)
        scaled_test = scaler.transform(relevant_test)
        relevant_train = pd.DataFrame(scaled_train, columns=relevant_train.columns, index=relevant_train.index)
        relevant_val = pd.DataFrame(scaled_val, columns=relevant_val.columns, index=relevant_val.index)
        relevant_test = pd.DataFrame(scaled_test, columns=relevant_test.columns, index=relevant_test.index)
        relevant_train.drop(columns="patient_id", inplace=True)
        relevant_val.drop(columns="patient_id", inplace=True)
        relevant_test.drop(columns="patient_id", inplace=True)

        window_generator = WindowGenerator(**self.window_generator_config)

        train_dataset, _ = self._create_dataset(relevant_train, patient_divisions_train, window_generator)
        if not train_dataset:
            logger.info(f"Not enough data for training, skipping {name}...")
            return pd.DataFrame()


        val_dataset, _ = self._create_dataset(relevant_val, patient_divisions_val, window_generator)
        if not val_dataset:
            logger.info(f"Not enough data for validation, skipping {name}...")
            return pd.DataFrame()

        test_dataset, patients_to_remove = self._create_dataset(relevant_test, patient_divisions_test, window_generator)
        if not test_dataset:
            logger.info(f"Not enough data for testing, skipping {name}...")
            return pd.DataFrame()
        self.deepant_config["name"] = name
        self.model[name] = DeepAnt(self.deepant_config, train_dataset, val_dataset, test_dataset, len(dataset_to_create["features"]), name)
        self.model[name].train()
        anomaly_dict = self.model[name].predict()

    @staticmethod
    def _create_dataset(data: pd.DataFrame, patient_divisions: dict, window_generator: WindowGenerator) -> (DataModule, list):
        patients_to_remove = []
        for patient_id, patient_info in patient_divisions.items():
            data = data[patient_info[0]:patient_info[1]]
            success = window_generator.split_data(data)
            if not success:
                patients_to_remove.append(patient_id)
        dataset = window_generator.generate_dataset()
        return dataset, patients_to_remove




    def _prepare_data(self, dataframe: pd.DataFrame) -> dict:
        patient_ids = list(dataframe["patient_id"].unique())
        np_patients = np.array(patient_ids)
        train_patients, remaining_patients = train_test_split(np_patients, test_size=1 - self.train_percentage,
                                                              random_state=self.sk_seed, shuffle=True)
        val_patients, test_patients = train_test_split(remaining_patients, test_size=self.test_percentage / (
                self.test_percentage + self.val_percentage), random_state=self.sk_seed, shuffle=True)
        train_data = dataframe[dataframe["patient_id"].isin(train_patients)].reset_index()
        val_data = dataframe[dataframe["patient_id"].isin(val_patients)].reset_index()
        test_data = dataframe[dataframe["patient_id"].isin(test_patients)].reset_index()
        result_dict = {"train": train_data, "val": val_data, "test": test_data}
        if not self.datasets_to_create:
            self.datasets_to_create = [{"name": column, "labels": [column], "features": [column]} for column in
                                       dataframe.columns if column not in self.columns_not_to_check]
        return result_dict

    def _handle_anomalies(self, anomalies: dict, anomalous_data : pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError




