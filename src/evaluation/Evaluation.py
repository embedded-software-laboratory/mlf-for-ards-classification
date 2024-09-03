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

    def evaluate_timeseries_models(self, timeseries_models: list[Model], cross_validation: bool, evaluation: bool,
                                   stage: str) -> Result:

        for timeseries_model in timeseries_models:
            model_evaluation = ModelEvaluation(self.config, timeseries_model, self)
            if stage == "Training":
                model_evaluation.evaluate(self.eval_info.dataset_training, stage)
            else:

                if cross_validation:
                    model_evaluation.cross_validate(self.eval_info.dataset_training)
                if evaluation:
                    model_evaluation.evaluate(self.eval_info.dataset_test, stage)
            model_result = ModelResultFactory.factory_method(model_evaluation.model_eval_info,
                                                             model_evaluation.evaluation_results)
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

    def evaluate(self, test_data, stage: str) -> None:
        feature_data = test_data.loc[:, test_data.columns != 'ards']
        if self.model.has_predict_proba():
            self.model_eval_info.predicted_probas_test = self.model.predict_proba(feature_data)[:, 1]

        else:
            self.model_eval_info.predicted_labels_test = self.model.predict(feature_data)

        self.model_eval_info.true_labels_test = test_data['ards']
        if self.config["process"]["perform_threshold_optimization"]:
            threshold_optimizers = self.config['evaluation']['threshold_optimization_algorithms']
        else:
            threshold_optimizers = ["Standard"]
        optimizer_list = []
        for optimizer in threshold_optimizers:
            eval_result = SplitFactory.factory_method(self.model_eval_info, f"{stage} split", optimizer, stage)
            optimizer_result = OptimizerFactory.factory_method([eval_result], optimizer)
            optimizer_list.append(optimizer_result)

        result = EvalResultFactory.factory_method(optimizer_list, stage)
        self.evaluation_results[stage] = result

    def cross_validate(self, data) -> None:
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

        for (train_set, test_set), i in zip(cross_validation.split(predictors, labels),
                                            range(self.evaluation.eval_info.n_splits)):

            predictors_train = predictors.iloc[train_set]
            labels_train = labels.iloc[train_set]
            predictors_test = predictors.iloc[test_set]
            labels_test = labels.iloc[test_set]

            # Learn model for the split
            self.model.train_model(predictors_train.assign(ards=labels_train))

            if self.config["process"]["save_models"]:
                save_path = self.config["storage_path"] if self.config["storage_path"] else "./Save/" + str(
                    datetime.now().strftime("%m-%d-%Y_%H-%M-%S")) + "/" + self.model.name + "_split_" + str(i)
                self.model.save_model(save_path)

            if self.model.has_predict_proba():
                self.model_eval_info.predicted_probas_test = self.model.predict_proba(predictors_test)[:, 1]
            else:
                self.model_eval_info.prediction_labels = self.model.predict(predictors_test)

            self.model_eval_info.true_labels_test = labels_test

            for optimizer in threshold_optimizers:
                eval_result = SplitFactory.factory_method(self.model_eval_info, f"Evaluation split {i}",
                                                          optimizer, "cross-validation")
                optimizer_eval_dict[optimizer].append(eval_result)
        optimizer_list = []
        for optimizer in threshold_optimizers:
            mean_split = SplitFactory.mean_split_factory_method(optimizer_eval_dict[optimizer])
            optimizer_eval_dict[optimizer].append(mean_split)
            complete_eval_list = optimizer_eval_dict[optimizer]
            optimizer_result = OptimizerFactory.factory_method(complete_eval_list, optimizer)
            optimizer_list.append(optimizer_result)
        result = EvalResultFactory.factory_method(optimizer_list, "Crossvalidation")
        self.evaluation_results["Crossvalidation"] = result

