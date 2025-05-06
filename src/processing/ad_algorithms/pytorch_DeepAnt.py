import math
import os
import sys
from multiprocessing import Pool
from typing import Any, Tuple, Optional
import pickle
import numpy as np
import pandas as pd

import logging
import joblib

import pytorch_lightning as pl
import torch.nn as  nn
import torch

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from processing.ad_algorithms.BaseAnomalyDetector import BaseAnomalyDetector
from processing.ad_algorithms.torchWindowGenerator import WindowGenerator, DataModule
from processing.ad_algorithms.torch_utils import check_device, check_directory, set_seed
from processing.datasets_metadata import AnomalyDetectionMetaData

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
        self.fc_input_size = math.ceil(size_after_pool2) * self.n_filters

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

    def predict_step(self, batch: Any, batch_idx: int, **kwargs) -> torch.Tensor:
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
                name (str): Name of the dataset.
        """

        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.feature_dim = feature_dim
        self.name = name
        self.max_processes = config["max_processes"]
        self.n_labels = len(config["labels"])


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
        if self.train_dataset is None:
            logger.error("Training dataset is None. Cannot train the model.")
            return
        if self.val_dataset is None:
            logger.error("Validation dataset is None. Cannot train the model.")
            return
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.config["batch_size"],
                                  shuffle=True,
        )
        logger.info(f"Training dataset size: {len(self.train_dataset)}")
        logger.info(f"Training batches: {len(train_loader)}")

        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.config["batch_size"],
                                shuffle=False,
                                
        )
        logger.info("Starting initial training...")
        self.initial_trainer.fit(self.anomaly_detector, train_loader, val_loader)
        initial_checkpoint_path = os.path.join(self.config["checkpoint_dir"], f"initial_model_{self.name}.ckpt")
        self.anomaly_detector = AnomalyDetector.load_from_checkpoint(
            checkpoint_path=initial_checkpoint_path,
            model=self.deepant_predictor,
            learning_rate=self.config["learning_rate"]
        )
        logger.info("Initial training completed. Starting main training...")
        self.trainer.fit(self.anomaly_detector, train_loader, val_loader)
        
        logger.info("Main training completed. Saving the best model...")

    def predict(self, test_dataset: DataModule = None) -> (int, dict):
        """
            Detects anomalies in the test dataset using the trained DeepAnt model.

            Returns:
                dict: Dictionary containing indices of detected anomalies for each feature.
        """
        
        if self.test_dataset is None and test_dataset is None:
            logger.error("Test dataset is None. Cannot detect anomalies.")
            return -1, {}
        if test_dataset is not None:
            dataset_to_use = test_dataset
        else:
            dataset_to_use = self.test_dataset
        logger.info("Starting detection of anomalies...")
        test_loader = DataLoader(dataset_to_use,
                                 batch_size=self.config["batch_size"],
                                 shuffle=False,
                                
        )
        try:
            best_model = AnomalyDetector.load_from_checkpoint(checkpoint_path=os.path.join(self.config["run_dir"], f"best_model_{self.name}.ckpt"),
                                                                model=self.deepant_predictor,
                                                                  lr=self.config["learning_rate"])
        except Exception as e:
            logger.error(f"Error loading the best model: {e}")
            return -2, {}
        all_predictions = []
        output = self.trainer.predict(best_model, test_loader)
        for item in output:
            batch_predictions = item.numpy().squeeze()
            all_predictions.append(batch_predictions)
        predictions = np.concatenate(all_predictions)

        ground_truth = test_loader.dataset.data_y.squeeze()
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
        return 0, anomalies_indices

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
        for feature_idx in range(self.n_labels):
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
        for feature_idx in range(self.n_labels):
            feature_scores = anomaly_scores[:, feature_idx]
            threshold = thresholds[feature_idx]
            anomalies = np.where(feature_scores > threshold)[0]
            logger.info(f"Identified {anomalies.shape} anomalies")
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
        self.windowed_data_disk_interaction = False
        self.loaded_data = False
        self.name = str(kwargs.get("name", "DeepAntDetector"))
        self.type = "DeepAnt"
        self.model = {}

        self._datasets_to_create = list(kwargs.get('dataset_to_create', []))
        self.val_percentage = float(kwargs.get('val_percentage', 0.1))
        self.train_percentage = float(kwargs.get('train_percentage', 0.2))
        self.test_percentage = 1 - self.val_percentage - self.train_percentage
        self.sk_seed = int(kwargs.get('seed', 42))
        set_seed(self.sk_seed)
        self.needs_full_data = True
        self.trainable = True
        self.window_generator_config = dict(kwargs.get('window_generator_config', {}))


        self.learning_rate = float(kwargs.get("learning_rate", 1e-3))
        self.hidden_size = int(kwargs.get("hidden_size", 256))
        self.max_epochs = int(kwargs.get("max_epochs", 100))
        self.max_initial_epochs = int(kwargs.get("max_initial_epochs", 10))
        self.batch_size = int(kwargs.get("batch_size", 2048))
        self.patience = int(kwargs.get("patience", 5))
        self.run_dir = str(kwargs.get("run_dir", "../Data/Models/AnomalyDetection/DeepAnt"))
        self.checkpoint_dir = str(kwargs.get("checkpoint_dir", "../Data/Models/AnomalyDetection/DeepAnt"))
        self.prepared_data_dir = str(kwargs.get("prepared_data_dir", "/work/rwth1474/Data/AnomalyDetection/windowed_data"))
        self.anomaly_data_dir = self.anomaly_data_dir = str(
            kwargs.get("anomaly_data_dir", f"/work/rwth1474/Data/AnomalyDetection/anomaly_data/SW_ABSAD_Mod/{self.name}"))
        check_directory(str(self.run_dir))
        check_directory(str(self.checkpoint_dir))
        check_directory(str(self.prepared_data_dir))
        check_directory(str(self.anomaly_data_dir))
        self.device = check_device()

        self.load_data =  bool(kwargs.get("load_data", True))
        self.save_data = bool(kwargs.get("save_data", True))
        self.retrain_models = dict(kwargs.get("retrain_models", {}))
        self._update_retrain_models()

        self.deepant_config = {
            "window_size" : int(self.window_generator_config.get("input_width", 10)),
            "prediction_size" : int(self.window_generator_config.get("output_width", 1)),
            "hidden_size" : int(self.hidden_size),
            "learning_rate" : float(self.learning_rate),
            "max_initial_epochs" : int(self.max_initial_epochs),
            "max_epochs" : int(self.max_epochs),
            "checkpoint_dir" : str(self.checkpoint_dir),
            "run_dir" : str(self.run_dir),
            "data_dir" : str(self.prepared_data_dir),
            "patience": int(self.patience),
            "device" : str(self.device),
            "batch_size" : int(self.batch_size),
            "max_processes" : int(self.max_processes)
        }

    @property
    def datasets_to_create(self) -> list:
        """
            Returns the datasets to create.
        """
        return self._datasets_to_create

    @datasets_to_create.setter
    def datasets_to_create(self, value: list):
        """
            Sets the datasets to create.
            Args:
                value (list): List of datasets to create.
        """
        if not isinstance(value, list):
            raise ValueError("datasets_to_create must be a list.")
        self._datasets_to_create = value
        self._update_retrain_models()


    def _update_retrain_models(self):
        for item in self._datasets_to_create:
            name = item["name"]
            if name not in self.retrain_models:
                self.retrain_models[name] = False




    def create_meta_data(self):
        meta_data_dict = super().create_meta_data()
        meta_data_dict["anomaly_detection_algorithm"] = self.type
        meta_data_dict["algorithm_specific_settings"] = {
            "train_percentage": self.train_percentage,
            "val_percentage": self.val_percentage,
            "test_percentage": self.test_percentage,
            "datasets": self.datasets_to_create,
            "seed": self.sk_seed,
            "checkpoint_dir": self.checkpoint_dir,
            "DeepAnt_config": self.deepant_config,
            "window_generator_config": self.window_generator_config,}

        if self.windowed_data_disk_interaction:
            meta_data_dict["algorithm_specific_settings"]["windowed_data_dir"] = self.prepared_data_dir


        return AnomalyDetectionMetaData(**meta_data_dict)



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

    def _handle_data_step(self, data: pd.DataFrame, dataset_to_create: dict, save_data: bool, load_data: bool, type_of_dataset: str) \
        -> (DataModule, list, pd.DataFrame):

        dataset = None
        patients_to_remove = []
        relevant = pd.DataFrame()
        if load_data or data is None:
            filename = self._get_filename_from_dataset_config(dataset_to_create, type_of_dataset)

            dataset, patients_to_remove, relevant = self._load_data(filename, type_of_dataset)
        if dataset is None and not data is None:
            dataset, patients_to_remove, relevant = self._prepare_data_step(data, dataset_to_create, save_data, type_of_dataset)



        return  dataset, patients_to_remove, relevant

    def _load_prepared_data(self, path: str, type_of_dataset: str) -> (DataModule, list, pd.DataFrame):
        """Returns none because data handling is done by the setup function"""
        return None



    def _prepare_data_step(self, data: pd.DataFrame, dataset_to_create: dict, save_data: bool, type_of_dataset: str)\
            -> (DataModule, list, pd.DataFrame):
        """
            Prepares the data for the anomaly detection step.

            Args:
                data (pd.DataFrame): The data from which to create the dataset.
                dataset_to_create (dict): Dictionary containing the dataset configuration.
                save_data (bool): Whether to save the data or not.
                type_of_dataset (str): Type of the dataset to create (train, val, test).

            Returns:
                DataModule: The prepared dataset.
                list: List of patients to remove (only relevant for test data).

        """


        name = dataset_to_create["name"]
        logger.info(f"Preparing data for {name}...")
        contained_patients = data["patient_id"].unique().tolist()
        relevant_columns = list(
            set(dataset_to_create["labels"] + dataset_to_create["features"] + ["patient_id", "time"]))
        relevant = data[relevant_columns]
        relevant = relevant.dropna(how='any', axis=0).reset_index(drop=True)



        patient_divisions = self._build_patient_dict(relevant)
        if len(relevant.index) == 0:
            logger.info(f"Not enough data for {type_of_dataset}, skipping {name}...")
            return None, [], pd.DataFrame()

        to_scale = relevant.copy(deep=True)
        to_scale["time"] = to_scale.groupby("patient_id")["time"].diff().fillna(0)
        if type_of_dataset == "train":
            scaler = MinMaxScaler()
            scaler.fit(to_scale)
            scaled = scaler.transform(to_scale)
            joblib.dump(scaler, os.path.join(self.prepared_data_dir + "/" + name + "_scaler.pkl"))
        else:
            try:
                scaler = joblib.load(os.path.join(self.prepared_data_dir + "/" + name + "_scaler.pkl"))
                scaled = scaler.transform(to_scale)
            except FileNotFoundError:
                logger.error(f"Scaler not found for {name}. Create a new dataset.")
                sys.exit(1)
        relevant_scaled = pd.DataFrame(scaled, columns=relevant_columns, index=to_scale.index)
        relevant_scaled.drop(columns="patient_id", inplace=True)


        specific_window_generator_config = self.window_generator_config.copy()
        specific_window_generator_config["feature_columns"] = dataset_to_create["features"]
        specific_window_generator_config["label_columns"] = dataset_to_create["labels"]
        window_generator = WindowGenerator(**specific_window_generator_config)

        dataset, patients_to_remove = self._create_dataset(relevant_scaled, patient_divisions, window_generator)
        contained_patients = [patient_id for patient_id in contained_patients if patient_id not in patients_to_remove]


        if not dataset:
            logger.info(f"Not enough data for {type_of_dataset}, skipping {name}...")
            return None, [], pd.DataFrame()

        if save_data:
            self.windowed_data_disk_interaction = True
            logger.info("Saving data")
            dataset_file_name = self._get_filename_from_dataset_config(dataset_to_create, type_of_dataset)
            feature_path = os.path.join(self.prepared_data_dir + "/" +  f"{dataset_file_name}_features.pkl")
            label_path = os.path.join(self.prepared_data_dir + "/" +f"{dataset_file_name}_labels.pkl")
            contained_patients_path = os.path.join(self.prepared_data_dir + "/" +f"{dataset_file_name}_contained_patients.pkl")
            self._save_file(dataset.data_x, feature_path, True)
            self._save_file(dataset.data_y, label_path, True)
            self._save_file(contained_patients, contained_patients_path, True)


            if type_of_dataset == "test":
                relevant = relevant[~relevant["patient_id"].isin(patients_to_remove)].reset_index(drop=True)
                logger.info(f"Number of entries for {name}: {len(relevant)}")
                patients_to_remove_path = os.path.join(self.prepared_data_dir + "/" + f"{dataset_file_name}_patients_to_remove.pkl")
                relevant_path = os.path.join(self.prepared_data_dir + "/" + f"{dataset_file_name}_relevant.pkl")
                self._save_file(patients_to_remove, patients_to_remove_path, True)
                self._save_file(relevant, relevant_path, True)


        if type_of_dataset != "test":
            return dataset, [], pd.DataFrame()
        else:
            return dataset, patients_to_remove, relevant

    def _load_data(self, name: str, type_of_dataset: str) -> (DataModule, list, pd.DataFrame):
        """
            Loads the data from the specified file.

            Args:
                name (str): The name of the dataset.
                type_of_dataset (str): Type of the dataset to load (train, val, test).

            Returns:
                DataModule: The loaded dataset.
                list: List of patients to remove (only relevant for test data).
        """

        try:
            with open(os.path.join(self.prepared_data_dir + "/" + name + f"_features.pkl"), "rb") as f:
                data_x = pickle.load(f)
            with open(os.path.join(self.prepared_data_dir + "/" + name + "_labels.pkl"), "rb") as f:
                data_y = pickle.load(f)
            if type_of_dataset == "test":
                patients_to_remove_path = os.path.join(self.prepared_data_dir + "/" + name + "_patients_to_remove.pkl")
                patients_to_remove_path = patients_to_remove_path.replace("test_", "")
                logger.info(f"Loading patients to remove from {patients_to_remove_path}")
                with open(patients_to_remove_path, "rb") as f:
                    patients_to_remove = pickle.load(f)
                with open(os.path.join(self.prepared_data_dir + "/" + name + "_relevant.pkl"), "rb") as f:
                    relevant_df = pickle.load(f)
            else:
                patients_to_remove = []
                relevant_df = pd.DataFrame()
            dataset = DataModule(data_x, data_y, device=self.device)
            self.windowed_data_disk_interaction = True
        except Exception as e:
            logger.error(f"Failed to load data for {name} {type_of_dataset}: {e}")
            dataset = None
            patients_to_remove = []
            relevant_df = pd.DataFrame()
        return dataset, patients_to_remove, relevant_df



    def _run_step(self, data: dict[str, pd.DataFrame], dataset_to_create: dict, retrain_model: bool, load_data: bool, save_data: bool) -> pd.DataFrame:
        """
            Runs the anomaly detection step for a specific dataset.

            Args:
                data (dict): Dictionary containing the prepared data.
                dataset_to_create (dict): Dictionary containing the dataset configuration.
                retrain_model (bool): If true we do not utilize the existing model, but train a new one.
                load_data (bool): Whether to load the data required for model training and anomaly detection from disk or create a new dataset. If loading fails, a new dataset is created.
                save_data (bool): Whether to save the data used for model training and anomaly detection or not.


            Returns:
                pd.DataFrame: The processed DataFrame after anomaly detection.
        """

        name = dataset_to_create["name"]
        train = data["train"]
        val = data["val"]
        test = data["test"]


        model_location = os.path.join(self.deepant_config["run_dir"], f"best_model_{name}.ckpt")
        logger.info(f"Check if model exists at location {str(model_location)}")
        model_exists = os.path.exists(os.path.join(self.deepant_config["run_dir"], f"best_model_{name}.ckpt"))
        logger.info(f"Model exists: {model_exists}")
        model_training = (not os.path.exists(model_location)) or retrain_model
        logger.info(f"Model training requested: {retrain_model}")
        logger.info(f"Model training : {model_training}")

        stages = []
        if model_training:
            stages.append("train")
            stages.append("val")
        stages.append("test")


        status, _, _, relevant_df =self.setup_deep_ant(dataset_to_create, stages, train, val, test, load_data, save_data)

        if (status == -1 and model_training) or status == -2:
            return pd.DataFrame()




        if model_training:
            self.model[name].train()

        anomaly_dict = self.model[name].predict()
        label_index_dict = {}
        for i in range(len(dataset_to_create["labels"])):
            label_index_dict[i] = dataset_to_create["labels"][i]
        anomaly_df = relevant_df[["patient_id", "time"]]
        marked_anomaly_dict = {}
        for key, value in anomaly_dict.items():
            index = int(key.replace("feature_", ""))
            name = label_index_dict[index]
            anomaly_indices = value.tolist()
            marked_anomaly_list = [False] * len(relevant_df)
            for i in anomaly_indices:
                marked_anomaly_list[i] = True
            marked_anomaly_dict[name + "_anomaly"] = marked_anomaly_list


        return anomaly_df



    


    def _create_dataset(self, data: pd.DataFrame, patient_divisions: dict, window_generator: WindowGenerator) -> (DataModule, list):
        # TODO add metadata
        patients_to_remove = []
        results = []
        patient_dfs = []
        for patient_id, patient_info in patient_divisions.items():
            patient_df = data[patient_info[0]:patient_info[1]]
            patient_dfs.append(patient_df)
        with Pool(processes=self.max_processes) as pool:
            results = pool.map(window_generator.split_data, patient_dfs)
        logger.info("Finished multiprocessing")
        for i in range(len(results)):
            if not results[i][2]:
                patients_to_remove.append(list(patient_divisions.keys())[i])
            else:
                data_x, data_y = results[i][0], results[i][1]
                window_generator.data_x.extend(data_x)
                window_generator.data_y.extend(data_y)
        dataset = window_generator.generate_dataset()
        window_generator.data_x = []
        window_generator.data_y = []

        return dataset, patients_to_remove



    def _prepare_data(self, dataframe: pd.DataFrame, save_data: bool = True, overwrite: bool = False ) -> dict:



        diagnosis = dataframe[["patient_id", "ards"]].dropna(subset=["ards"]).reset_index(drop=True)

        train_patients, remaining_patients = train_test_split(diagnosis, test_size=1 - self.train_percentage,
                                                              random_state=self.sk_seed, shuffle=True, stratify=diagnosis["ards"])
        val_patients, test_patients = train_test_split(remaining_patients, test_size=self.test_percentage / (
                self.test_percentage + self.val_percentage), random_state=self.sk_seed, shuffle=True, stratify=remaining_patients["ards"])

        train_patient_ids = train_patients["patient_id"].unique().tolist()
        val_patient_ids = val_patients["patient_id"].unique().tolist()
        test_patient_ids = test_patients["patient_id"].unique().tolist()
        train_data = dataframe[dataframe["patient_id"].isin(train_patient_ids)].reset_index()
        val_data = dataframe[dataframe["patient_id"].isin(val_patient_ids)].reset_index()
        test_data = dataframe[dataframe["patient_id"].isin(test_patient_ids)].reset_index()
        data_dict = {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }


        if not self.datasets_to_create:
            self.datasets_to_create = [{"name": column, "labels": [column], "features": [column, "time"]} for column in
                                       dataframe.columns if column not in self.columns_not_to_check]

        if "prepare" in self.active_stages:
            datatypes_to_prepare = ["train", "val", "test"]
        elif "train" in self.active_stages:
            datatypes_to_prepare = ["train", "val"]
        elif "predict" in self.active_stages:
            datatypes_to_prepare = ["test"]
        else:
            return {}


        for item in self.datasets_to_create:
            self._prepare_data_multi(datatypes_to_prepare, data_dict, item, overwrite)
        #with Pool(processes=self.max_processes) as pool:
        #    pool.starmap(self._prepare_data_multi , [(datatypes_to_prepare, data_dict, item, overwrite) for item in self.datasets_to_create])


        return data_dict

    def _create_meta_data_preparation(self, test_data: pd.DataFrame) -> dict:
        contained_patients = test_data["patient_id"].unique().tolist()
        meta_data_dict = {
            "algorithm_specific_settings": {
                "window_generator_config": self.window_generator_config,
                "train_percentage": self.train_percentage,
                "val_percentage": self.val_percentage,
                "test_percentage": 1 - self.val_percentage - self.train_percentage,
                "seed": self.sk_seed
            },
            "datasets": self.datasets_to_create,
            "contained_patients": contained_patients,
        }
        return meta_data_dict


    def _prepare_data_multi(self, datatypes_to_prepare, data_dict: dict, item,  overwrite: bool = False) -> None:

        for dataset_type in datatypes_to_prepare:
            self._prepare_dataset(data_dict,  item, dataset_type, overwrite)

    def _prepare_dataset(self, data_dict: dict, item: dict, dataset_type: str, overwrite: bool) -> None:
        feature_file = os.path.join(self.prepared_data_dir, f"{item['name']}_{dataset_type}_features.pkl")
        label_file = os.path.join(self.prepared_data_dir, f"{item['name']}_{dataset_type}_labels.pkl")

        if not os.path.exists(feature_file) or not os.path.exists(label_file) or overwrite:
            self._prepare_data_step(data_dict[dataset_type], item, True, dataset_type)






    def setup_deep_ant(self, dataset_to_create: dict, stages: list[str],  data_train: pd.DataFrame, data_val: pd.DataFrame,
                       data_test: pd.DataFrame, load_data: bool = False, save_data: bool = True) -> (int, Optional[DataModule], list, pd.DataFrame):


        train_dataset = None
        val_dataset = None
        test_dataset = None

        patients_to_remove = []
        relevant_data = pd.DataFrame()

        for stage in stages:
            if stage == "train":
                train_dataset, _, _ = self._handle_data_step(data_train, dataset_to_create, save_data, load_data, stage)
                if not train_dataset:
                    logger.info(f"Not enough data for {stage}, skipping {dataset_to_create['name']}...")
                    return -1,  [], pd.DataFrame()
            elif stage == "val":
                val_dataset, _, _ = self._handle_data_step(data_val, dataset_to_create, save_data, load_data, stage)
                if not val_dataset:
                    logger.info(f"Not enough data for {stage}, skipping {dataset_to_create['name']}...")
                    return -1,  [], pd.DataFrame()
            elif stage == "test":
                test_dataset, patients_to_remove, relevant_data = self._handle_data_step(data_test, dataset_to_create, save_data, load_data, stage)
                if not test_dataset:
                    logger.info(f"Not enough data for {stage}, skipping {dataset_to_create['name']}...")
                    return -2, None, [], pd.DataFrame()
        name = dataset_to_create["name"]
        self.deepant_config["name"] = name
        self.deepant_config["labels"] = dataset_to_create["labels"]
        feature_dim = 0
        if "train" in stages and "val" in stages:
            
            train_dim = train_dataset.data_x[0].shape[1]
            val_dim = val_dataset.data_x[0].shape[1]
            if train_dim != val_dim:
                logger.info(f"Train and val data dimensions do not match for {name}, skipping...")
                return -1, [], pd.DataFrame()
            feature_dim = train_dim
        elif "test" in stages:
            test_dim = test_dataset.data_x[0].shape[1]
            feature_dim = test_dim
        self.model[name] = DeepAnt(self.deepant_config, train_dataset, val_dataset, test_dataset,
                                   feature_dim, name)
        return 0, patients_to_remove, relevant_data

    @staticmethod
    def _get_dataset_config_from_file_name(filename: str) -> dict:
        split = filename.split("_")
        dataset_name = split[0]
        dataset_labels = split[2].split("+")
        dataset_features = split[1].split("+")
        return {
            "name": dataset_name,
            "labels": dataset_labels,
            "features": dataset_features
        }

    @staticmethod
    def _get_filename_from_dataset_config(dataset_to_create: dict, type_of_dataset: str) -> str:

        name_str = dataset_to_create["name"]
        features_str = "+".join(dataset_to_create["features"])
        labels_str = "+".join(dataset_to_create["labels"])
        filename_str = f"{name_str}_{features_str}_{labels_str}_{type_of_dataset}"
        return filename_str


    def _train_ad_model(self, data_training: pd.DataFrame, data_validation: pd.DataFrame, **kwargs):
        stages = ["train", "val"]
        if not self.datasets_to_create:
            if not data_training is None and not data_training.empty:
                self.datasets_to_create = [
                    {"name": column,
                     "labels": [column],
                     "features": [column]}
                    for column in list(data_training.columns) if column not in self.columns_not_to_check]
            else:
                contained_files = os.listdir(self.prepared_data_dir)
                contained_test = [file.removesuffix("_test_features.pkl") for file in contained_files if file.endswith("test_features.pkl")]
                contained_val = [file.removesuffix("_val_features.pkl") for file in contained_files if file.endswith("val_features.pkl")]
                contained_train = [file.removesuffix("_train_features.pkl") for file in contained_files if file.endswith("train_features.pkl")]
                all_present = list(set(contained_train).intersection(set(contained_val)).intersection(set(contained_test)))
                for filename in all_present:
                    self.datasets_to_create.append(self._get_dataset_config_from_file_name(filename))
        for dataset in self.datasets_to_create:
            name = dataset["name"]
            retrain_model = self.retrain_models.get(name, True)
            model_location = os.path.join(self.deepant_config["run_dir"], f"best_model_{name}.ckpt")
            logger.info(f"Check if model exists at location {str(model_location)}")
            model_exists = os.path.exists(os.path.join(self.deepant_config["run_dir"], f"best_model_{name}.ckpt"))
            logger.info(f"Model exists: {model_exists}")
            model_training = (not os.path.exists(model_location)) or retrain_model
            logger.info(f"Model training requested: {retrain_model}")
            logger.info(f"Model training : {model_training}")
            logger.info(f"Training {dataset['name']}...")
            if model_training:
                status, _, _ =self.setup_deep_ant(dataset, stages, data_training, data_validation, None,  load_data=True, save_data=False)
                if status == -1:
                    logger.info(f"Not enough data for {dataset['name']}, skipping...")
                    continue
                self._train_ad_model_step(dataset)

    def _set_dataset_according_to_available_models(self):
        datasets_without_models = []
        for dataset in self.datasets_to_create:
            name = dataset["name"]
            model_name = f"best_model_{name}.ckpt"
            model_location = os.path.join(self.deepant_config["run_dir"], model_name)
            if not os.path.exists(model_location):
                datasets_without_models.append(name)
        self.datasets_to_create = [dataset for dataset in self.datasets_to_create if dataset not in datasets_without_models]

    def _predict(self, dataframe: pd.DataFrame, **kwargs ) -> dict:
        if not self.datasets_to_create:
            contained_files = os.listdir(self.prepared_data_dir)
            contained_test = [file.removesuffix("_test_features.pkl") for file in contained_files if
                              file.endswith("test_features.pkl")]
            for name in contained_test:
                self.datasets_to_create.append(self._get_dataset_config_from_file_name(name))
            self._set_dataset_according_to_available_models()
        anomaly_df = pd.DataFrame()
        relevant_df = pd.DataFrame()
        anomaly_df_list = []
        relevant_df_list = []

        for dataset in self.datasets_to_create:
            anomaly_df_list, relevant_df_list = self._predict_step(dataset, dataframe, anomaly_df_list, relevant_df_list)

        for i in range(len(anomaly_df_list)):
            if i == 0:
                anomaly_df = anomaly_df_list[i]
            else:
                anomaly_df = pd.merge(anomaly_df, anomaly_df_list[i], how="outer", on=["patient_id", "time"])
        for i in range(len(relevant_df_list)):
            if i == 0:
                relevant_df = relevant_df_list[i]
            else:
                relevant_df = pd.merge(relevant_df_list[i], relevant_df_list[i], how="outer", on=["patient_id", "time"])


        anomaly_df.fillna(False, inplace=True)
        self._save_anomaly_df(anomaly_df)
        anomaly_count_dict = self._calculate_anomaly_count(anomaly_df, relevant_df)


        return {
            "anomaly_df": anomaly_df,
            "anomaly_count": anomaly_count_dict,
        }

    def _predict_step(self, dataset: dict, dataframe: pd.DataFrame, anomaly_df_list: list[pd.DataFrame], relevant_df_list: list[pd.DataFrame]) -> (list[pd.DataFrame], list[pd.DataFrame]):
        anomaly_df_dataset = pd.DataFrame()
        name = dataset["name"]
        status, patients_to_remove, relevant_data = self.setup_deep_ant(dataset, ["test"], None, None, dataframe,
                                                                        load_data=True, save_data=True)
        if status != 0:
            logger.info(f"Not enough data for {dataset['name']}, skipping...")
            return anomaly_df_list, relevant_df_list
        status, anomaly_indices = self.model[name].predict()
        if status != 0:
            logger.info(f"No model found for {dataset['name']}, skipping...")
            return anomaly_df_list, relevant_df_list
        anomaly_df_dataset["patient_id"] = relevant_data["patient_id"]
        anomaly_df_dataset["time"] = relevant_data["time"]
        anomaly_list = []
        for i in range(len(anomaly_df_dataset)):
            if i in anomaly_indices:
                anomaly_list.append(True)
            else:
                anomaly_list.append(False)
        for label in dataset["labels"]:
            anomaly_df_dataset[label] = anomaly_list
        anomaly_df_list.append(anomaly_df_dataset)
        relevant_df_list.append(relevant_data)
        return anomaly_df_list, relevant_df_list








    def _train_ad_model_step(self, dataset_to_create: dict):


        name = dataset_to_create["name"]
        self.model[name].train()





