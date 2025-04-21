import math
import os
import sys
from multiprocessing import Pool
from typing import Any, Tuple
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

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
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

        initial_checkpoint_path = os.path.join(self.config["run_dir"], f"initial_model_{self.name}.ckpt")
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
                                 shuffle=False,
                                
        )
        best_model = AnomalyDetector.load_from_checkpoint(checkpoint_path=os.path.join(self.config["run_dir"], f"best_model_{self.name}.ckpt"),
                                                            model=self.deepant_predictor,
                                                              lr=self.config["learning_rate"])
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
        self.window_generator_config = dict(kwargs.get('window_generator_config', {}))


        self.learning_rate = float(kwargs.get("learning_rate", 1e-3))
        self.hidden_size = int(kwargs.get("hidden_size", 256))
        self.max_epochs = int(kwargs.get("max_epochs", 100))
        self.max_initial_epochs = int(kwargs.get("max_initial_epochs", 10))
        self.batch_size = int(kwargs.get("batch_size", 2048))
        self.patience = int(kwargs.get("patience", 5))
        self.run_dir = str(kwargs.get("run_dir", "../Data/Models/AnomalyDetection/DeepAnt"))
        self.checkpoint_dir = str(kwargs.get("checkpoint_dir", "../Data/Models/AnomalyDetection/DeepAnt"))
        self.windowed_data_dir = str(kwargs.get("data_dir", "/work/rwth1474/Data/AnomalyDetection/windowed_data"))
        self.anomaly_data_dir = str(kwargs.get("anomaly_data_dir", "../Data/AnomalyDetection/anomaly_data/DeepAnt"))
        check_directory(str(self.run_dir))
        check_directory(str(self.checkpoint_dir))
        check_directory(str(self.windowed_data_dir))
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
            "data_dir" : str(self.windowed_data_dir),
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

    def prepare_full_data_for_storage(self, data: pd.DataFrame) -> None:
        """
            Prepares the full data for storage.

            Args:
                data (pd.DataFrame): The data to prepare.
        """
        data = self._prepare_data(data)
        train = data["train"]
        val = data["val"]
        test = data["test"]
        for item in self.datasets_to_create:
            logger.info(f"Preparing data for {item['name']}")
            self._prepare_data_step(train, item, True, "train")
            self._prepare_data_step(val, item, True, "val")
            self._prepare_data_step(test, item, True, "test")



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
            results.append(self._run_step(prepared_data, item, self.retrain_models[item["name"]], self.load_data, self.save_data))

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
        with open(os.path.join(self.anomaly_data_dir, f"anomaly_data_{self.name}.pkl"), "wb") as f:
            pickle.dump(df_anomaly, f)
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
        relevant_columns = list(
            set(dataset_to_create["labels"] + dataset_to_create["features"] + ["patient_id", "time"]))
        relevant = data[relevant_columns]
        relevant = relevant.dropna(how='any', axis=0).reset_index(drop=True)


        patient_divisions = self._build_patient_dict(relevant)

        if type_of_dataset == "train":
            scaler = MinMaxScaler()
            scaler.fit(relevant)
            scaled = scaler.transform(relevant)
            joblib.dump(scaler, os.path.join(self.windowed_data_dir + "/" + name + "_scaler.pkl"))
        else:
            try:
                scaler = joblib.load(os.path.join(self.windowed_data_dir + "/" + name + "_scaler.pkl"))
                scaled = scaler.transform(relevant)
            except FileNotFoundError:
                logger.error(f"Scaler not found for {name}. Create a new dataset.")
                sys.exit(1)
        relevant_scaled = pd.DataFrame(scaled, columns=relevant_columns, index=relevant.index)
        relevant_scaled.drop(columns="patient_id", inplace=True)


        specific_window_generator_config = self.window_generator_config.copy()
        specific_window_generator_config["feature_columns"] = dataset_to_create["features"]
        specific_window_generator_config["label_columns"] = dataset_to_create["labels"]
        window_generator = WindowGenerator(**specific_window_generator_config)

        dataset, patients_to_remove = self._create_dataset(relevant_scaled, patient_divisions, window_generator)


        if not dataset:
            logger.info(f"Not enough data for {type_of_dataset}, skipping {name}...")
            return None, []

        if save_data:
            logger.info("Saving data")
            with open(os.path.join(self.windowed_data_dir + "/" + name + "_" + type_of_dataset + "_features.pkl"), "wb") as f:
                pickle.dump(dataset.data_x, f)
            with open(os.path.join(self.windowed_data_dir + "/" + name + "_" + type_of_dataset + "_labels.pkl"), "wb") as f:
                pickle.dump(dataset.data_y, f)

            if type_of_dataset == "test":
                relevant = relevant[~relevant["patient_id"].isin(patients_to_remove)].reset_index(drop=True)
                logger.info(f"Number of entries for {name}: {len(relevant)}")
                with open(os.path.join(self.windowed_data_dir + "/" + name + "_patients_to_remove.pkl"), "wb") as f:
                    pickle.dump(patients_to_remove, f)
                with open(os.path.join(self.windowed_data_dir + "/" + name + "_relevant.pkl"), "wb") as f:
                    pickle.dump(relevant, f)


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
            with open(os.path.join(self.windowed_data_dir + "/" + name + f"_{type_of_dataset}_features.pkl"), "rb") as f:
                data_x = pickle.load(f)
            with open(os.path.join(self.windowed_data_dir + "/" + name + f"_{type_of_dataset}_labels.pkl"), "rb") as f:
                data_y = pickle.load(f)
            if type_of_dataset == "test":
                with open(os.path.join(self.windowed_data_dir + "/" + name + "_patients_to_remove.pkl"), "rb") as f:
                    patients_to_remove = pickle.load(f)
                with open(os.path.join(self.windowed_data_dir + "/" + name + "_relevant.pkl"), "rb") as f:
                    relevant_df = pickle.load(f)
            else:
                patients_to_remove = []
                relevant_df = pd.DataFrame()
            dataset = DataModule(data_x, data_y, device=self.device)
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




        train_dataset = None
        val_dataset = None
        model_location = os.path.join(self.deepant_config["run_dir"], f"best_model_{name}.ckpt")
        logger.info(f"Check if model exists at location {str(model_location)}")
        model_exists = os.path.exists(os.path.join(self.deepant_config["run_dir"], f"best_model_{name}.ckpt"))
        logger.info(f"Model exists: {model_exists}")
        model_training = (not os.path.exists(model_location)) or retrain_model
        
        logger.info(f"Model training requested: {retrain_model}")
        
        logger.info(f"Model training : {model_training}")
        if load_data:

            if model_training:
                train_dataset, _, _ = self._load_data(name, "train")
                if not train_dataset:
                    train_dataset, _, _ = self._prepare_data_step(train, dataset_to_create, True, "train")
                    if not train_dataset:
                        return pd.DataFrame()

                val_dataset, _, _ = self._load_data(name, "val")
                if not val_dataset:
                    val_dataset, _, _ = self._prepare_data_step(val, dataset_to_create, True, "val")
                    if not val_dataset:
                        return pd.DataFrame()
            test_dataset, patients_to_remove, relevant_df = self._load_data(name, "test")
            if not test_dataset:
                test_dataset, patients_to_remove, relevant_df = self._prepare_data_step(test, dataset_to_create, True,
                                                                           "test")
                if not test_dataset:
                    return pd.DataFrame()

        else:

            if model_training:
                train_dataset, _, _ = self._prepare_data_step(train, dataset_to_create, save_data, "train")
                if not train_dataset:
                    return pd.DataFrame()

                val_dataset, _, _ = self._prepare_data_step(val, dataset_to_create, save_data, "val")
                if not val_dataset:
                    return pd.DataFrame()
            test_dataset, patients_to_remove, relevant_df = self._prepare_data_step(test, dataset_to_create, save_data, "test")
            if not test_dataset:
                return pd.DataFrame()
        self.deepant_config["name"] = name
        self.deepant_config["labels"] = dataset_to_create["labels"]
        self.model[name] = DeepAnt(self.deepant_config, train_dataset, val_dataset, test_dataset,
                                   len(dataset_to_create["features"]), name)

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
        for key, value in marked_anomaly_dict.items():
            anomaly_df[key] = value
        test = test.merge(anomaly_df, how="left", on=["patient_id", "time"]).fillna(False)
        return test






    def _create_dataset(self, data: pd.DataFrame, patient_divisions: dict, window_generator: WindowGenerator) -> (DataModule, list):
        # TODO add metadata
        patients_to_remove = []
        counter = 0
        results = []
        patient_dfs = []
        for patient_id, patient_info in patient_divisions.items():
            patient_df = data[patient_info[0]:patient_info[1]]
            patient_dfs.append(patient_df)
        with Pool(processes=self.max_processes) as pool:
            results = pool.map(window_generator.split_data, patient_dfs)
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




    def _prepare_data(self, dataframe: pd.DataFrame) -> dict:
        # TODO add which patients are in which set


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
        result_dict = {"train": train_data, "val": val_data, "test": test_data}

        if not self.datasets_to_create:
            self.datasets_to_create = [{"name": column, "labels": [column], "features": [column, "time"]} for column in
                                       dataframe.columns if column not in self.columns_not_to_check]
        return result_dict

    def _handle_anomalies(self, anomalies: dict, anomalous_data : pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError




