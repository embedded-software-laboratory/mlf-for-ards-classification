from processing.datasets import Dataset
from ml_models.model_interface import Model
from metrics.models.Generic_Models import *
from sklearn.metrics import roc_curve
from metrics.Metrics import *
from metrics.ThresholdOptimizer import GeometricRoot, MaxTPR, MaxTPRMinFPR, Standard

from pydantic import BaseModel, ValidationInfo, field_validator, ConfigDict


class EvaluationInformation:
    def __init__(self, config, model, dataset_training=None, dataset_test=None):

        self.model = model
        self.eval_storage_location = config['storage_path']  # TODO check what is correct
        self.dataset_training = dataset_training  # TODO replace by representation of dataset
        self.dataset_test = dataset_test  # TODO replace by representation of dataset
        self.model_name = model.name
        self.eval_name = config['evaluation']['evaluation_name']
        self.cross_validation_performed = config["process"]["perform_cross_validation"]
        self.evaluation_performed = config["process"]["calculate_evaluation_metrics"]
        if self.cross_validation_performed:

            self.n_splits = config["evaluation"]["cross_validation"]["n_splits"]
            self.shuffle = config["evaluation"]["cross_validation"]["shuffle"]
            self.random_state = config["evaluation"]["cross_validation"]["random_state"]
        else:
            self.n_splits = 0
        
        self.model_has_proba = model.has_predict_proba()
        self.requested_metrics = config["evaluation"]["evaluation_metrics"]
        self.contained_metrics = []  # TODO check how to communicate incompatible metrics
        for metric in self.requested_metrics:
            if not eval(metric + "().needs_probabilities()"):
                self.contained_metrics.append(metric)

            if self.model_has_proba and eval(metric + "().needs_probabilities()"):
                self.contained_metrics.append(metric)

        if self.model_has_proba:
            self.threshold_optimizers = config["evaluation"]["threshold_optimization_algorithms"]
        else:
            self.threshold_optimizers = ["Standard"]

        self.predicted_labels = None
        self.predicted_probas = None
        self.true_labels = None


class Result(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    result_name: str
    storage_location: str

    training_dataset: object = None  # TODO add data set information
    test_dataset: object = None  # TODO add data set information

    used_model_type: Model
    used_model_name: str = None
    contained_optimizers: dict[str, GenericThresholdOptimization]

    crossvalidation_performed: bool
    crossvalidation_random_state: int = None
    crossvalidation_shuffle: bool = None
    crossvalidation_splits: int = None
    evaluation_performed: bool

    @field_validator('crossvalidation_random_state', 'crossvalidation_splits')
    @classmethod
    def check_crossvalidation_settings_int(cls, v: int, info: ValidationInfo):
        if info.data['crossvalidation_performed']:
            if isinstance(v, int):
                assert v is not None, f'{info.field_name} must be set if crossvalidation_performed is set to True'
                assert v >= 0, f'{info.field_name} must be greater than zero if crossvalidation_performed is set to False'
        return v

    @field_validator('crossvalidation_shuffle', )
    @classmethod
    def check_crossvalidation_shuffle_settings_bool(cls, v: bool, info: ValidationInfo):
        if info.data['crossvalidation_performed']:
            if isinstance(v, bool):
                assert v is not None, f'{info.field_name} must be set if crossvalidation_performed is set to True'
        return v


class ResultFactory:
    @staticmethod
    def factory_method(evaluation: EvaluationInformation, optimizer_list: list[GenericThresholdOptimization]) \
            -> Result:
        result_name = evaluation.eval_name
        used_model_name = evaluation.model_name
        used_model_type = evaluation.model
        cross_validation_performed = evaluation.cross_validation_performed
        n_splits = None
        shuffle = None
        random_state = None
        if cross_validation_performed:
            n_splits = evaluation.n_splits
            shuffle = evaluation.shuffle
            random_state = evaluation.random_state
        evaluation_performed = evaluation.evaluation_performed
        dict_optimizer = {}
        for optimizer in optimizer_list:
            dict_optimizer[optimizer.name] = optimizer
        storage_location = evaluation.eval_storage_location

        return Result(result_name=result_name, storage_location=storage_location,
                      training_dataset=evaluation.dataset_training, test_dataset=evaluation.dataset_test,
                      used_model_type=used_model_type, used_model_name=used_model_name,
                      contained_optimizers=dict_optimizer, crossvalidation_performed=cross_validation_performed,
                      cross_validation_random_state=random_state, cross_validation_shuffle=shuffle,
                      cross_validation_splints=n_splits, evaluation_performed=evaluation_performed)


class SplitFactory:
    @staticmethod
    def factory_method(evaluation: EvaluationInformation, split_name: str, optimizer_name: str) -> GenericSplit:
        contained_metrics_dict = {}
        optimizer = eval(optimizer_name + "()")
        metric_information = {"prediction_probs": evaluation.predicted_probas,
                              "predicted_label": evaluation.predicted_labels,
                              "true_labels": evaluation.true_labels,
                              "calc_func": optimizer.calculate_optimal_threshold}

        if evaluation.model_has_proba:
            fpr, tpr, thresholds = roc_curve(evaluation.true_labels, evaluation.predicted_probas)
            metric_information["fpr"] = fpr
            metric_information["tpr"] = tpr
            metric_information["thresholds"] = thresholds
            optimal_threshold = OptimalProbability().calculate_metric(metric_information)
            prediction_labels = (evaluation.predicted_probas > optimal_threshold).astype(int)
        else:
            prediction_labels = evaluation.predicted_labels
        metric_information["prediction_labels"] = prediction_labels
        metric_information["true_labels"] = evaluation.true_labels

        for metric in evaluation.contained_metrics:
            metric_obj = eval(metric + "()")
            contained_metrics_dict[metric] = metric_obj.calculate_metric(metric_information)

        return GenericSplit(split_name=split_name, contained_metrics=contained_metrics_dict)


class MeanSplitFactory:
    @staticmethod
    def factory_method(splits: list[GenericSplit]) -> GenericSplit:
        metric_dict = {}

        for split in splits:
            for metric in split.contained_metrics:
                if metric.name not in metric_dict:
                    metric_dict[metric.name] = [metric]
                else:
                    metric_dict[metric.name].append(metric)
        for metric_name, metric_list in metric_dict.items:
            average_metric = eval("metric_list[0].metric_spec().calculate_metric_mean(metric_list)")
            metric_dict[metric_name] = average_metric
        return GenericSplit(split_name="mean", contained_metrics=metric_dict)


class OptimizerFactory:
    @staticmethod
    def factory_method(splits: list[GenericSplit], optimizer_name) -> GenericThresholdOptimization:
        contained_splits = {}
        for split in splits:
            contained_splits[split.split_name] = split
        return GenericThresholdOptimization(contained_splits=contained_splits, optimizer_name=optimizer_name)
