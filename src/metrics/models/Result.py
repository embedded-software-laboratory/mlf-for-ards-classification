from processing.datasets import Dataset
from models.model_interface import Model
from metrics.models.Generic_Models import *
from sklearn.metrics import roc_curve
from metrics.Metrics import *


class EvaluationInformation:
    def __init__(self, config, model, dataset_training, dataset_test, model_name):
        self.model = model
        self.storage_location = config['storage_location'] # TODO check what is correct
        self.dataset_training = dataset_training
        self.dataset_test = dataset_test
        self.model_name = model_name


        if config["process"]["perform_cross_validation"]:
            self.n_splits = config["evaluation"]["cross_validation"]["n_splits"]
            self.shuffle = config["evaluation"]["cross_validation"]["shuffle"]
        else:
            self.n_splits = 0

        self.model_has_proba = model.has_predict_proba()
        self.requested_metrics = config["evaluation_metrics"]
        self.contained_metrics = []
        for metric in self.requested_metrics:
            if not eval(metric + "().needs_probabilities()"):
                self.contained_metrics.append(metric)

            if self.model_has_proba and eval(metric + "().needs_probabilities()"):
                self.contained_metrics.append(metric)

        if self.model_has_proba:
            self.threshold_optimizers = config["evaluation"]["threshold_optimizers"]
        else:
            self.threshold_optimizers = ["Standard"]

        self.predicted_labels = None
        self.predicted_probas = None
        self.true_labels = None


class Result(BaseModel):
    result_name: str
    storage_location: str

    training_dataset: Dataset
    test_dataset: Dataset

    used_model_type: Model
    used_model_name: str = None
    contained_optimizers: GenericThresholdOptimization


class ResultFactory:

    def factory_method(self, optimizer_list: list[GenericThresholdOptimization], evaluation: EvaluationInformation) \
            -> Result:
        return Result()


class SplitFactory:
    def factory_method(self, evaluation: EvaluationInformation, split_name: str, optimizer_name: str) -> GenericSplit:
        contained_metrics_dict = {}
        metric_information = {"prediction_probs": evaluation.predicted_probas,
                              "prediction_labels": evaluation.predicted_labels,
                              "true_labels": evaluation.true_labels,
                              "calc_func": eval(optimizer_name + "().calculate_optimal_threshold")}

        if evaluation.model_has_proba:
            fpr, tpr, thresholds = roc_curve(evaluation.true_labels, evaluation.predicted_probas)
            metric_information["fpr"] = fpr
            metric_information["tpr"] = tpr
            metric_information["thresholds"] = thresholds
            optimal_threshold = OptimalProbability.calculate_metric(metric_information)
            prediction_labels = (evaluation.predicted_probas[:, 1] > optimal_threshold).astype(int)
        else:
            prediction_labels = evaluation.predicted_labels
        metric_information["prediction_labels"] = prediction_labels
        metric_information["true_labels"] = evaluation.true_labels

        for metric in evaluation.contained_metrics:
            metric_obj = eval(metric + "()")
            contained_metrics_dict[metric] = metric_obj.calculate_metric(metric_information)

        return GenericSplit(split_name=split_name, contained_metrics=contained_metrics_dict)


class MeanSplitFactory:
    def factory_method(self, splits: list[GenericSplit]) -> GenericSplit:
        metric_dict = {}

        for split in splits:
            for metric in split.contained_metrics:
                if metric.name not in metric_dict:
                    metric_dict[metric.name] = [metric]
                else:
                    metric_dict[metric.name].append(metric)
        for metric_name, metric_list in metric_dict.items():
            average_metric = eval("metric_list[0].metric_spec().calculate_metric_mean(metric_list)")
            metric_dict[metric_name] = average_metric
        return GenericSplit(split_name="mean", contained_metrics=metric_dict)


class OptimizerFactory:
    def factory_method(self, splits: list[GenericSplit], optimizer_name) -> GenericThresholdOptimization:
        contained_splits = {}
        for split in splits:
            contained_splits[split.split_name] = split
        return GenericThresholdOptimization(contained_splits=contained_splits, optimizer_name=optimizer_name)
