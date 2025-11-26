from ml_models.timeseries_model import TimeSeriesProbaModel
from sklearn import svm
from sklearn.utils.validation import check_X_y, check_array
import pickle
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class SupportVectorMachineModel(TimeSeriesProbaModel):

    def __init__(self):
        super().__init__()
        self.name = "Support Vector Machine"
        self.algorithm = "SupportVectorMachine"

        self.hyperparameters = {
            "C": 1.0,
            "kernel": 'linear',
            "degree": 3,
            "gamma": 'scale',
            "coef0": 0.0,
            "shrinking": True,
            "probability": True,
            "tol": 0.001,
            "cache_size": 1000,
            "class_weight": None,
            "verbose": False,
            "max_iter": -1,
            "decision_function_shape": 'ovr',
            "break_ties": False,
            "random_state": 42
        }

        # model instance and metadata about training features
        self.model = self._init()
        self.feature_names = None

    def _init(self):
        # Initialize SVC with given hyperparameters (pass into constructor)
        return svm.SVC(**self.hyperparameters)

    def _prepare_training_data(self, training_data: pd.DataFrame):
        if "ards" not in training_data.columns:
            raise ValueError("training_data must contain 'ards' column as label")

        # Candidate feature columns to drop if present
        drop_cols = [c for c in ["ards", "patient_id", "time", "timestamp", "identifier"] if c in training_data.columns]
        X_df = training_data.drop(columns=drop_cols, errors='ignore').copy()

        # Keep only numeric columns (SVM requires numeric features). Log if non-numeric cols are dropped.
        non_numeric = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            logger.debug(f"Dropping non-numeric columns for SVM training: {non_numeric}")
            X_df = X_df.select_dtypes(include=[np.number])

        y = pd.to_numeric(training_data["ards"], errors="coerce")

        # Align and drop rows with NaN in y or in X
        valid_mask = y.notna() & X_df.notna().all(axis=1)
        X_df = X_df.loc[valid_mask]
        y = y.loc[valid_mask]

        if X_df.empty or y.empty:
            raise ValueError("No valid training rows after filtering NaNs / non-numeric features")

        # store feature names for predict time
        self.feature_names = X_df.columns.tolist()
        X = X_df.values
        y = y.values

        # basic validation
        X_checked, y_checked = check_X_y(X, y, ensure_2d=True)
        return X_checked, y_checked

    def train_model(self, training_data):
        """Train the SVM on provided training_data DataFrame."""
        logger.info("Training Support Vector Machine model...")
        X, y = self._prepare_training_data(training_data)

        # create a fresh model with current hyperparameters (in case they changed)
        self.model = svm.SVC(**self.hyperparameters)
        self.model.fit(X, y)
        self.trained = True
        logger.info("SVM training complete. Model marked as trained.")

    def _prepare_predict_X(self, data):
        # Accept DataFrame or numpy array
        if isinstance(data, pd.DataFrame):
            if self.feature_names is None:
                raise RuntimeError("Model has no stored feature names - train the model first")
            missing = [f for f in self.feature_names if f not in data.columns]
            if missing:
                raise ValueError(f"Input is missing required features: {missing}")
            X_df = data[self.feature_names].copy()
            # ensure numeric
            X_df = X_df.select_dtypes(include=[np.number])
            if X_df.shape[1] != len(self.feature_names):
                raise ValueError("Some features are non-numeric or missing in input DataFrame")
            X = X_df.values
        else:
            # assume array-like
            X = np.asarray(data)
        X_checked = check_array(X, ensure_2d=True)
        return X_checked

    def predict(self, data):
        """Return class predictions for input data."""
        if not self.trained:
            raise RuntimeError("Model not trained")
        X = self._prepare_predict_X(data)
        return self.model.predict(X)

    def predict_proba(self, data):
        """Return class probabilities if available."""
        if not self.trained:
            raise RuntimeError("Model not trained")
        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError("Underlying SVM model does not support predict_proba (set probability=True)")
        X = self._prepare_predict_X(data)
        return self.model.predict_proba(X)

    def get_params(self):
        return self.hyperparameters.copy()

    def set_params(self, params: dict):
        # update hyperparameters dict and re-init model if necessary
        for key, value in params.items():
            if key in self.hyperparameters:
                self.hyperparameters[key] = value
        # reinitialize model with new params
        self.model = svm.SVC(**self.hyperparameters)

    def save_model(self, filepath):
        fullpath = filepath + f"{self.algorithm}_{self.name}.pkl"
        with open(fullpath, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "hyperparameters": self.hyperparameters
            }, f)
        logger.debug(f"Model saved to {fullpath}")

    def load_model(self, filepath):
        fullpath = filepath + f"{self.algorithm}_{self.name}.pkl"
        with open(fullpath, "rb") as f:
            state = pickle.load(f)
        self.model = state.get("model")
        self.feature_names = state.get("feature_names")
        self.hyperparameters = state.get("hyperparameters", self.hyperparameters)
        self.trained = True if self.model is not None else False
        logger.debug(f"Model loaded from {fullpath}")

    def has_predict_proba(self):
        return hasattr(self.model, "predict_proba")
