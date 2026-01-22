from __future__ import annotations

import numpy as np

from metrics.Metrics import *
from metrics.ThresholdOptimizer import *

from processing import TimeseriesMetaData, TimeSeriesMetaDataManagement
from processing.datasets_metadata import ImageMetaData
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluation import ModelEvaluationInformation

from sklearn.metrics import roc_curve
import logging

logger = logging.getLogger(__name__)


class EvalResultFactory:
    """ Contains the result of a single evaluation run"""

    @staticmethod
    def factory_method(optimizer_list: list[GenericThresholdOptimization], training_set_meta_data: TimeseriesMetaData, test_set_meta_data:TimeseriesMetaData,
                       evaltype: str, crossvalidation_performed: bool = False, crossvalidation_random_state: int =None,
                          crossvalidation_shuffle:bool= None, crossvalidation_splits: int =None, evaluation_performed:bool=False) -> EvalResult:
        logger.info(f"Creating EvalResult for type '{evaltype}' (crossvalidation_performed={crossvalidation_performed})")
        eval_name = evaltype

        dict_optimizer = {}
        for optimizer in optimizer_list:
            dict_optimizer[optimizer.optimization_name] = optimizer
        logger.debug(f"Contained optimizers: {list(dict_optimizer.keys())}")

        if crossvalidation_performed:
            result = EvalResult(eval_type=eval_name, training_dataset=training_set_meta_data, test_dataset=training_set_meta_data,
                                contained_optimizers=dict_optimizer, crossvalidation_performed=crossvalidation_performed,
                                crossvalidation_random_state=crossvalidation_random_state,crossvalidation_shuffle=crossvalidation_shuffle,
                                crossvalidation_splits=crossvalidation_splits, evaluation_performed=evaluation_performed
                                )
            logger.info("EvalResult (cross-validation) created successfully")
            return result
        else:
            result = EvalResult(eval_type=eval_name, training_dataset=training_set_meta_data, test_dataset=test_set_meta_data,
                                contained_optimizers=dict_optimizer, crossvalidation_performed=crossvalidation_performed,
                                evaluation_performed=evaluation_performed
                                )
            logger.info("EvalResult (single evaluation) created successfully")
            return result

    @staticmethod
    def from_dict(eval_dict: dict) -> EvalResult:
        logger.info("Reconstructing EvalResult from dictionary")
        contained_optimizers = {}
        for optimizer in eval_dict["contained_optimizers"]:
             contained_optimizers[optimizer] = OptimizerFactory.from_dict(eval_dict["contained_optimizers"][optimizer])

        # Handle both TimeseriesMetaData and ImageMetaData
        def load_metadata(metadata_dict):
            if metadata_dict is None:
                return None
            # Check if it's ImageMetaData by looking for disease_type field
            if 'disease_type' in metadata_dict:
                return ImageMetaData(**metadata_dict)
            else:
                return TimeSeriesMetaDataManagement.load_from_dict(metadata_dict)

        content = {
            "eval_type": eval_dict["eval_type"],
            "training_dataset": load_metadata(eval_dict.get("training_dataset")),
            "test_dataset": load_metadata(eval_dict.get("test_dataset")),
            "contained_optimizers": contained_optimizers,
            "crossvalidation_performed": eval_dict["crossvalidation_performed"],
            "crossvalidation_random_state": eval_dict["crossvalidation_random_state"],
            "crossvalidation_shuffle": eval_dict["crossvalidation_shuffle"],
            "crossvalidation_splits": eval_dict["crossvalidation_splits"],
            "evaluation_performed": eval_dict["evaluation_performed"],
            }
        logger.debug(f"EvalResult reconstructed: eval_type={content['eval_type']}, optimizers={list(contained_optimizers.keys())}")
        return EvalResult(**content)


class ModelResultFactory:
    """Contains the result of multiple evaluations runs for a specified model"""

    @staticmethod
    def factory_method(evaluation: ModelEvaluationInformation, contained_evals: dict, training_evaluation: EvalResult,
                       stage: str) -> ModelResult:
        logger.info(f"Building ModelResult for model '{evaluation.model_name}' at stage '{stage}'")
        if evaluation.model.storage_location is None:
            evaluation.model.storage_location = "Unknown"
            logger.debug("Model storage location was None, set to 'Unknown'")

        if stage == "Evaluation":
            contained_evals["Training"] = training_evaluation
            logger.debug("Attached Training EvalResult to contained_evals under key 'Training'")

        model_result = ModelResult(used_model_location=evaluation.model.storage_location, used_model_name=evaluation.model.name,
                           contained_evals=contained_evals, used_model_algorithm=evaluation.model.algorithm, used_model_type=evaluation.model.type)
        logger.info(f"ModelResult built for model '{evaluation.model_name}'")
        return model_result


class ResultManagement:
    """Contains the result of multiple models which may have multiple evaluation runs"""

    def factory_method(self, factory_type: str, ingredients: dict) -> ExperimentResult:
        logger.info(f"ResultManagement.factory_method called with type '{factory_type}'")
        if factory_type == "new":
            exp_result = self._factory_method_new_result(ingredients["EvaluationInformation"], ingredients["model_results"])
            logger.info("New ExperimentResult constructed")
        else:
            logger.error("Unsupported factory_type passed to ResultManagement.factory_method")
            raise ValueError("Factory type must be Evaluation")
        return exp_result

    def merge(self, content: dict, merge_type: str) -> ExperimentResult:
        logger.info(f"ResultManagement.merge called with merge_type '{merge_type}'")
        if merge_type == "CV_EVAL":
            return self._merge_cv_eval_results(content["CV"], content["Eval"], content["storage_location"], content["eval_name"])
        else:
            logger.error("Unsupported merge_type passed to ResultManagement.merge")
            raise ValueError("Merge type must be CV_EVAL")

    @staticmethod
    def _factory_method_new_result(evaluation: EvaluationInformation, model_results: dict) -> ExperimentResult:
        logger.debug(f"Creating ExperimentResult with name '{evaluation.experiment_name}' and storage '{evaluation.eval_storage_location}'")
        return ExperimentResult(result_name=evaluation.experiment_name, storage_location=evaluation.eval_storage_location,
                                contained_model_results=model_results)

    @staticmethod
    def _factory_merge_result(result_info: dict) -> ExperimentResult:
        logger.debug("Constructing ExperimentResult from merged result_info")
        return ExperimentResult(**result_info)

    def _merge_cv_eval_results(self, cv_result: ExperimentResult, eval_result: ExperimentResult, storage_location: str, eval_name: str) -> ExperimentResult:
        logger.info("Merging cross-validation and evaluation ExperimentResults")
        contained_models = set()
        model_eval_dict = {}
        model_algorithm_dict = {}
        model_storage_dict = {}

        for model_name in cv_result.contained_model_results.keys():
            contained_models.add(model_name)

        for model_name in eval_result.contained_model_results.keys():
            contained_models.add(model_name)

        for model_name in list(contained_models):
            model_eval_dict[model_name] = {}

            if model_name in eval_result.contained_model_results:
                for key in eval_result.contained_model_results[model_name].contained_evals.keys():
                    model_eval_dict[model_name][key] = eval_result.contained_model_results[model_name].contained_evals[
                        key]

                model_algorithm_dict[model_name] = eval_result.contained_model_results[model_name].used_model_algorithm
                model_storage_dict[model_name] = eval_result.contained_model_results[model_name].used_model_location
            if model_name in cv_result.contained_model_results:
                for key in cv_result.contained_model_results[model_name].contained_evals.keys():
                    model_eval_dict[model_name][key] = cv_result.contained_model_results[model_name].contained_evals[
                        key]
                if model_name not in eval_result.contained_model_results.keys():
                    model_algorithm_dict[model_name] = cv_result.contained_model_results[
                        model_name].used_model_algorithm
                    model_storage_dict[model_name] = cv_result.contained_model_results[model_name].used_model_location
        model_result_dict = {}
        for model_name in model_eval_dict.keys():
            contained_evals = model_eval_dict[model_name]

            model_algorithm = model_algorithm_dict[model_name]
            model_storage = model_storage_dict[model_name]

            model_result = ModelResult(used_model_location=model_storage, used_model_name=model_name,
                                       contained_evals=contained_evals, used_model_type="TimeSeriesModel",
                                       used_model_algorithm=model_algorithm)
            model_result_dict[model_name] = model_result
        ingredients = {"result_name": eval_name, "storage_location": storage_location,
                       "contained_model_results": model_result_dict}
        logger.info("Merged ExperimentResult constructed")
        return self._factory_merge_result(ingredients)


class SplitFactory:

    @staticmethod
    def mean_split_factory_method(splits: list[GenericSplit]) -> GenericSplit:
        logger.debug("Computing mean split from list of splits")
        metric_dict = {}

        for split in splits:
            for _, metric in split.contained_metrics.items():
                if metric.metric_name not in metric_dict:
                    metric_dict[metric.metric_name] = [metric]
                else:
                    metric_dict[metric.metric_name].append(metric)
        for metric_name, metric_list in metric_dict.items():
            average_value = metric_list[0].metric_spec.calculate_metric_mean(metric_list)
            average_metric = metric_list[0].metric_spec.create_from_value(average_value, metric_list[0].metric_name)
            metric_dict[metric_name] = average_metric
            logger.debug(f"Mean metric computed for {metric_name}: {average_value}")
        logger.info("Mean split created")
        return GenericSplit(split_name="mean", contained_metrics=metric_dict)


    @staticmethod
    def factory_method(evaluation: ModelEvaluationInformation, split_name: str, optimizer_name: str, stage: str) \
            -> GenericSplit:
        logger.debug(f"Creating GenericSplit for split '{split_name}', optimizer '{optimizer_name}', stage '{stage}'")
        contained_metrics_dict = {}
        optimizer = eval(optimizer_name + "()")
        metric_information = {"prediction_probs": evaluation.predicted_probas_test,
                              "predicted_label": evaluation.predicted_labels_test,
                              "true_labels": evaluation.true_labels_test,
                              "calc_func": optimizer.calculate_optimal_threshold}
        if evaluation.model_has_proba:
            logger.debug("Preparing ROC information for probability-based evaluation")
            logger.warning(type(evaluation.true_labels_test))
            logger.warning(type(evaluation.predicted_probas_test))
            logger.warning(np.unique(evaluation.true_labels_test))
            logger.warning(np.array(evaluation.predicted_probas_test).shape)
            evaluation.true_labels_test = np.asarray(evaluation.true_labels_test).astype(int)
            evaluation.predicted_probas_test = np.asarray(evaluation.predicted_probas_test)
            fpr, tpr, thresholds = roc_curve(evaluation.true_labels_test, evaluation.predicted_probas_test)
            metric_information["fpr"] = fpr
            metric_information["tpr"] = tpr
            metric_information["thresholds"] = thresholds
            if stage != "Training":
                logger.debug("Using training optimal threshold for evaluation stage")
                training_eval = evaluation.model.training_evaluation.contained_optimizers
                training_result = training_eval[optimizer_name].contained_splits["Training split"]
                optimal_threshold_training = training_result.contained_metrics["OptimalProbability"].metric_value
                prediction_labels = (evaluation.predicted_probas_test > optimal_threshold_training).astype(int)
            else:
                optimal_threshold = OptimalProbability().calculate_metric(metric_information, stage)
                prediction_labels = (evaluation.predicted_probas_test > optimal_threshold).astype(int)
        else:
            logger.debug("Model does not provide probabilities; using predicted labels directly")
            prediction_labels = evaluation.predicted_labels_test

        metric_information["predicted_label"] = prediction_labels
        metric_information["true_labels"] = evaluation.true_labels_test

        for metric in evaluation.contained_metrics:
            metric_obj = eval(metric + "()")
            contained_metrics_dict[metric] = metric_obj.calculate_metric(metric_information, stage)
        logger.info(f"GenericSplit '{split_name}' created with {len(contained_metrics_dict)} metrics")
        return GenericSplit(split_name=split_name, contained_metrics=contained_metrics_dict)

    @staticmethod
    def from_dict(split_dict: dict) -> GenericSplit:
        logger.debug(f"Reconstructing GenericSplit from dict: {split_dict.get('split_name','unknown')}")
        contained_metrics = {}

        for metric in split_dict["contained_metrics"]:
            contained_metrics[metric] = eval(metric + "()").create_from_dict(split_dict["contained_metrics"][metric])

        logger.info(f"GenericSplit '{split_dict.get('split_name','unknown')}' reconstructed")
        return GenericSplit(split_name=split_dict["split_name"], contained_metrics=contained_metrics)


class OptimizerFactory:
    @staticmethod
    def factory_method(splits: list[GenericSplit], optimizer_name) -> GenericThresholdOptimization:
        logger.debug(f"Building GenericThresholdOptimization for optimizer '{optimizer_name}'")
        contained_splits = {}
        for split in splits:
            contained_splits[split.split_name] = split
        logger.info(f"Optimizer '{optimizer_name}' constructed with splits: {list(contained_splits.keys())}")
        return GenericThresholdOptimization(contained_splits=contained_splits, optimization_name=optimizer_name)

    @staticmethod
    def from_dict(optimizer_dict: dict) ->  GenericThresholdOptimization:
        logger.debug(f"Reconstructing GenericThresholdOptimization from dict: {optimizer_dict.get('optimization_name','unknown')}")
        contained_splits = {}
        for split in optimizer_dict["contained_splits"]:
            contained_splits[split] = SplitFactory.from_dict(optimizer_dict["contained_splits"][split])

        logger.info(f"GenericThresholdOptimization '{optimizer_dict.get('optimization_name','unknown')}' reconstructed")
        return GenericThresholdOptimization(contained_splits=contained_splits, optimization_name=optimizer_dict["optimization_name"])
