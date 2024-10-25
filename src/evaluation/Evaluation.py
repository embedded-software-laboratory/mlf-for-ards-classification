import copy
from datetime import datetime
from typing import Union

import pandas as pd
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, matthews_corrcoef, roc_curve, make_scorer
from sklearn.model_selection import StratifiedKFold
from metrics.Models import EvalResult
from metrics.ModelFactories import *
from evaluation.EvaluationInformation import ModelEvaluationInformation, EvaluationInformation
from ml_models.model_interface import Model

import os


class Evaluation:
    def __init__(self, config, dataset_training, dataset_test):
        self.config = config
        self.eval_info = EvaluationInformation(config, dataset_training, dataset_test)
        self.model_results = {}

    def evaluate_timeseries_models(self, timeseries_models: list[Model]) -> ExperimentResult:

        for timeseries_model in timeseries_models:
            if not timeseries_model.trained:
                print(f"Skipping {timeseries_model.name} because it makes no sense to evaluate an untrained model")
                continue
            else:
                model_evaluation = ModelEvaluation(self.config, timeseries_model, self)
                predictors = self.eval_info.dataset_test.loc[:, self.eval_info.dataset_test.columns != 'ards']
                labels = self.eval_info.dataset_test['ards']
                model_evaluation.evaluate_timeseries_model(predictors, labels, "Evaluation")
                model_result = ModelResultFactory.factory_method(model_evaluation.model_eval_info,
                                                                model_evaluation.evaluation_results,
                                                                model_evaluation.model.training_evaluation,
                                                                "Evaluation")
                self.model_results[timeseries_model.name] = model_result

        overall_result = ResultFactory.factory_method(self.eval_info, self.model_results)
        return overall_result

    def cross_validate_timeseries_models(self, timeseries_models: list[Model]) -> ExperimentResult:

        for timeseries_model in timeseries_models:
            model_evaluation = ModelEvaluation(self.config, timeseries_model, self)
            model_evaluation.cross_validate_timeseries_model(self.eval_info.dataset_training)
            model_result = ModelResultFactory.factory_method(model_evaluation.model_eval_info, model_evaluation.evaluation_results,
                                                             model_evaluation.model.training_evaluation, stage="CrossValidation")
            self.model_results[timeseries_model.name] = model_result

        overall_result = ResultFactory.factory_method(self.eval_info, self.model_results)
        return overall_result


class ModelEvaluation:

    def __init__(self, config, model, evaluation: Union[Evaluation, None]):
        self.evaluation = evaluation
        self.config = config
        self.model = model
        self.model_eval_info = ModelEvaluationInformation(config, model)
        self.evaluation_results = {}

    def evaluate_timeseries_model(self, predictors, true_labels, stage: str, split_name: str =" split") -> None:

        eval_split_name = stage + split_name

        if self.model.has_predict_proba():
            self.model_eval_info.predicted_probas_test = self.model.predict_proba(predictors)[:, 1]

        else:
            self.model_eval_info.predicted_labels_test = self.model.predict(predictors)

        self.model_eval_info.true_labels_test = true_labels
        if self.config["process"]["perform_threshold_optimization"]:
            threshold_optimizers = self.config['evaluation']['threshold_optimization_algorithms']
        else:
            threshold_optimizers = ["Standard"]
        optimizer_list = []
        for optimizer in threshold_optimizers:
            eval_result = SplitFactory.factory_method(self.model_eval_info, eval_split_name, optimizer, stage)
            optimizer_result = OptimizerFactory.factory_method([eval_result], optimizer)
            optimizer_list.append(optimizer_result)

        result = EvalResultFactory.factory_method(optimizer_list, stage)

        self.evaluation_results[stage] = result

    def cross_validate_timeseries_model(self, data) -> None:
        if self.evaluation.eval_info is None:
            print("Can not perform cross validation without evaluation information")
            return
        labels = data["ards"]
        predictors = data.loc[:, data.columns != 'ards']

        # Create Splits for Crossvalidation
        cross_validation = StratifiedKFold(n_splits=self.evaluation.eval_info.n_splits,
                                           shuffle=self.evaluation.eval_info.shuffle,
                                           random_state=self.evaluation.eval_info.random_state)

        if self.config["process"]["perform_threshold_optimization"]:
            threshold_optimizers = self.config['evaluation']['threshold_optimization_algorithms']
        else:
            threshold_optimizers = ["Standard"]

        optimizer_eval_dict = {}
        for optimizer in threshold_optimizers:
            optimizer_eval_dict[optimizer] = []
        split_training_evaluation_dict = {}
        for (train_set, test_set), i in zip(cross_validation.split(predictors, labels),
                                            range(self.evaluation.eval_info.n_splits)):

            predictors_train = predictors.iloc[train_set]
            labels_train = labels.iloc[train_set]
            training_data = predictors_train.assign(ards=labels_train)

            predictors_test = predictors.iloc[test_set]
            labels_test = labels.iloc[test_set]
            test_data = predictors_test.assign(ards=labels_test)

            # Learn model for the split
            self.model.train_timeseries(training_data, self.config, "Training")
            training_eval = self.model.training_evaluation

            if self.config["process"]["save_models"]:
                save_path = self.config["storage_path"] if self.config["storage_path"] else "./Save/" + str(
                    datetime.now().strftime("%m-%d-%Y_%H-%M-%S")) + "/" + self.model.name + "_split_" + str(i)
                self.model.save_model(save_path)

            if self.model.has_predict_proba():
                self.model_eval_info.predicted_probas_test = self.model.predict_proba(predictors_test)[:, 1]
                self.model_eval_info.predicted_probas_training = self.model.predict_proba(predictors_train)[:, 1]
            else:
                self.model_eval_info.predicted_labels_test = self.model.predict(predictors_test)
                self.model_eval_info.predicted_labels_training = self.model.predict(predictors_train)

            self.model_eval_info.true_labels_test = labels_test
            self.model_eval_info.true_labels_training = labels_train

            for optimizer in threshold_optimizers:
                training_result = copy.deepcopy(training_eval.contained_optimizers[optimizer].contained_splits["Training split"])
                
                eval_result = SplitFactory.factory_method(self.model_eval_info, f"CrossValidationEvaluation split {i}",
                                                          optimizer, "Evaluation")
                training_result.split_name = f"CrossValidationTraining split: {i}"
                optimizer_eval_dict[optimizer].append(training_result)
                optimizer_eval_dict[optimizer].append(eval_result)
        optimizer_list = []
        for optimizer in threshold_optimizers:
            mean_split = SplitFactory.mean_split_factory_method(optimizer_eval_dict[optimizer])
            optimizer_eval_dict[optimizer].append(mean_split)
            complete_eval_list = optimizer_eval_dict[optimizer]
            optimizer_result = OptimizerFactory.factory_method(complete_eval_list, optimizer)
            optimizer_list.append(optimizer_result)
        result = EvalResultFactory.factory_method(optimizer_list, "CrossValidation")
        self.evaluation_results["CrossValidation"] = result

