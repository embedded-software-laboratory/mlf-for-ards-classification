import logging

from ml_models.model_interface import Model
from processing import TimeSeriesDataset

from metrics.Metrics import *

logger = logging.getLogger(__name__)


class EvaluationInformation:
    """
    Holds global evaluation configuration and dataset references used by the evaluation pipeline.
    Encapsulates cross-validation settings and flags that control which evaluation steps are executed.
    """

    def __init__(self, config, dataset_training: TimeSeriesDataset=None, dataset_test: TimeSeriesDataset=None):
        logger.info("Initializing EvaluationInformation...")
        self.dataset_training = dataset_training
        self.dataset_test = dataset_test
        self.experiment_name = config['evaluation']['evaluation_name']
        self.eval_storage_location = config['storage_path']
        self.cross_validation_performed = config["process"]["perform_cross_validation"]
        self.evaluation_performed = config["process"]["calculate_evaluation_metrics"]

        logger.debug(f"Experiment name: {self.experiment_name}")
        logger.debug(f"Storage location: {self.eval_storage_location}")
        logger.debug(f"Cross validation enabled: {self.cross_validation_performed}")
        logger.debug(f"Evaluation metrics enabled: {self.evaluation_performed}")

        if self.cross_validation_performed:
            self.n_splits = config["evaluation"]["cross_validation"]["n_splits"]
            self.shuffle = config["evaluation"]["cross_validation"]["shuffle"]
            self.random_state = config["evaluation"]["cross_validation"]["random_state"]
            logger.info(f"Cross-validation settings - n_splits: {self.n_splits}, shuffle: {self.shuffle}, random_state: {self.random_state}")
        else:
            self.n_splits = 0
            self.shuffle = False
            self.random_state = None
            logger.debug("Cross-validation not performed; n_splits set to 0.")


class ModelEvaluationInformation:
    """
    Container for information required to evaluate a single model instance.
    Stores configuration of requested metrics, whether the model provides probabilities,
    and placeholders for predictions and true labels for both test and training splits.
    """

    def __init__(self, config, model: Model, dataset_training=None, dataset_test=None):
        logger.info(f"Initializing ModelEvaluationInformation for model: {getattr(model, 'name', 'unknown')}")
        self.model = model
        self.model_name = model.name

        self.model_has_proba = model.has_predict_proba()
        logger.debug(f"Model supports probability output: {self.model_has_proba}")

        self.requested_metrics = config["evaluation"]["evaluation_metrics"]
        self.contained_metrics = []  # filters incompatible metrics based on model capabilities
        logger.debug(f"Requested metrics: {self.requested_metrics}")

        for metric in self.requested_metrics:
            try:
                metric_instance = eval(metric + "()")
                needs_proba = metric_instance.needs_probabilities()
                if not needs_proba:
                    self.contained_metrics.append(metric)
                if self.model_has_proba and needs_proba:
                    self.contained_metrics.append(metric)
                logger.debug(f"Metric '{metric}': needs_proba={needs_proba} -> included={metric in self.contained_metrics}")
            except Exception as e:
                logger.warning(f"Could not evaluate metric '{metric}': {e}")

        if self.model_has_proba:
            self.threshold_optimizers = config["evaluation"]["threshold_optimization_algorithms"]
        else:
            self.threshold_optimizers = ["Standard"]
        logger.debug(f"Threshold optimizers set to: {self.threshold_optimizers}")

        # Prediction/result placeholders populated during evaluation or cross-validation
        self.predicted_labels_test = None
        self.predicted_probas_test = None
        self.true_labels_test = None

        self.predicted_labels_training = None
        self.predicted_probas_training = None
        self.true_labels_training = None

        logger.info(f"ModelEvaluationInformation initialized for model: {self.model_name} with metrics: {self.contained_metrics}")


