from evaluation.EvaluationInformation import ModelEvaluationInformation, EvaluationInformation
from metrics.Models import GenericThresholdOptimization, EvalResult, GenericSplit
from metrics.Metrics import OptimalProbability

from metrics.ThresholdOptimizer import *
from metrics.Metrics import *
import json

from sklearn.metrics import roc_curve


class EvalResultFactory:
    """ Contains the result of a single evaluation run"""

    @staticmethod
    def factory_method(evaluation: EvaluationInformation, optimizer_list: list[GenericThresholdOptimization],
                       evaltype: str) -> EvalResult:
        eval_name = evaltype
        training_dataset = evaluation.dataset_training
        test_dataset = evaluation.dataset_test
        
        cross_validation_performed = evaluation.cross_validation_performed
        cross_validation_random_state = evaluation.random_state
        cross_validation_shuffle = evaluation.shuffle
        cross_validation_splits = evaluation.n_splits
        evaluation_performed = evaluation.evaluation_performed
        dict_optimizer = {}
        for optimizer in optimizer_list:
            dict_optimizer[optimizer.optimization_name] = optimizer
        return EvalResult(eval_type=eval_name, training_dataset=training_dataset, test_dataset=test_dataset,
                          contained_optimizers=dict_optimizer,
                          crossvalidation_performed=cross_validation_performed,
                          crossvalidation_random_state=cross_validation_random_state,
                          crossvalidation_shuffle=cross_validation_shuffle,
                          crossvalidation_splits=cross_validation_splits, evaluation_performed=evaluation_performed)


class ModelResultFactory:
    """Contains the result of multiple evaluations runs for a specified model"""

    @staticmethod
    def factory_method(evaluation: ModelEvaluationInformation, contained_evals: dict) -> ModelResult:

        return ModelResult(used_model_location=evaluation.model.storage_location, used_model_name=evaluation.model_name,
                           contained_evals=contained_evals)


class ResultFactory:
    """Contains the result of multiple models which may have multiple evaluation runs"""

    @staticmethod
    def factory_method(evaluation: EvaluationInformation, model_results: dict) -> Result:

        return Result(result_name=evaluation.experiment_name, storage_location=evaluation.eval_storage_location,
                      contained_model_results=model_results)


class ResultFactoryOld:
    @staticmethod
    def factory_method(evaluation: ModelEvaluationInformation, optimizer_list: list[GenericThresholdOptimization]) \
            -> EvalResult:
        result_name = evaluation.experiment_name
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
        
        storage_location = evaluation.eval_storage_location
        
        return EvalResult(result_name=result_name, storage_location=storage_location,
                          training_dataset=evaluation.dataset_training, test_dataset=evaluation.dataset_test,
                          used_model_type=used_model_type, used_model_name=used_model_name,
                          contained_optimizers=dict_optimizer, crossvalidation_performed=cross_validation_performed,
                          cross_validation_random_state=random_state, cross_validation_shuffle=shuffle,
                          cross_validation_splints=n_splits, evaluation_performed=evaluation_performed)


class SplitFactory:
    @staticmethod
    def factory_method(evaluation: ModelEvaluationInformation, split_name: str, optimizer_name: str) -> GenericSplit:
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
        metric_information["predicted_label"] = prediction_labels
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
            for _, metric in split.contained_metrics.items():
                if metric.metric_name not in metric_dict:
                    metric_dict[metric.metric_name] = [metric]
                else:
                    metric_dict[metric.metric_name].append(metric)
        for metric_name, metric_list in metric_dict.items():
            average_metric = metric_list[0].metric_spec.calculate_metric_mean(metric_list)
            metric_dict[metric_name] = average_metric
        return GenericSplit(split_name="mean", contained_metrics=metric_dict)


class OptimizerFactory:
    @staticmethod
    def factory_method(splits: list[GenericSplit], optimizer_name) -> GenericThresholdOptimization:
        contained_splits = {}
        for split in splits:
            contained_splits[split.split_name] = split
        return GenericThresholdOptimization(contained_splits=contained_splits, optimization_name=optimizer_name)
