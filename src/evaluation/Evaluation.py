

import copy

from typing import Union
from datetime import datetime


from evaluation.EvaluationInformation import EvaluationInformation, ModelEvaluationInformation

from processing import TimeSeriesDataset, TimeSeriesDatasetManagement, TimeSeriesMetaDataManagement
from ml_models import Model
from metrics.ModelFactories import ExperimentResult, ModelResultFactory, ResultManagement, SplitFactory, OptimizerFactory, EvalResultFactory

from sklearn.model_selection import StratifiedKFold







class Evaluation:
    def __init__(self, config, dataset_training: TimeSeriesDataset, dataset_test: TimeSeriesDataset):
        self.config = config
        self.eval_info = EvaluationInformation(config, dataset_training, dataset_test)
        self.model_results = {}

    def evaluate_timeseries_models(self, models_to_evaluate_dict: dict[str, list[Model]]) -> ExperimentResult:
        print("Evaluating models")
        for model_algorithm in models_to_evaluate_dict:
            print(f"Evaluating models for algorithm {model_algorithm}")
            for timeseries_model in models_to_evaluate_dict[model_algorithm]:
                print(f"Evaluating {timeseries_model.name}")
                if not timeseries_model.trained:
                    print(f"Skipping {timeseries_model.name} because it makes no sense to evaluate an untrained model")
                    continue
                else:
                    model_evaluation = ModelEvaluation(self.config, timeseries_model, self)
                    predictors = self.eval_info.dataset_test.content.loc[:, self.eval_info.dataset_test.content.columns != 'ards']
                    meta_data_training_set = self.eval_info.dataset_training.meta_data
                    meta_data_test_set = self.eval_info.dataset_test.meta_data
                    labels = self.eval_info.dataset_test.content['ards']
                    model_evaluation.evaluate_timeseries_model(predictors, labels, "Evaluation", meta_data_training_set, meta_data_test_set)
                    model_result = ModelResultFactory.factory_method(model_evaluation.model_eval_info,
                                                                    model_evaluation.evaluation_results,
                                                                    model_evaluation.model.training_evaluation,
                                                                    "Evaluation")
                    self.model_results[timeseries_model.name] = model_result
        ingredients = {"EvaluationInformation": self.eval_info, "model_results": self.model_results}
        overall_result = ResultManagement().factory_method("new", ingredients)
        return overall_result

    def cross_validate_timeseries_models(self, models_to_cross_validate_dict: dict[str, list[Model]]) -> ExperimentResult:

        for model_algorithm in models_to_cross_validate_dict:
            for timeseries_model in models_to_cross_validate_dict[model_algorithm]:
                model_evaluation = ModelEvaluation(self.config, timeseries_model, self)
                model_evaluation.cross_validate_timeseries_model(self.eval_info.dataset_training)
                model_result = ModelResultFactory.factory_method(model_evaluation.model_eval_info, model_evaluation.evaluation_results,
                                                                 model_evaluation.model.training_evaluation, stage="CrossValidation")
                self.model_results[timeseries_model.name] = model_result
        ingredients = {"EvaluationInformation": self.eval_info, "model_results": self.model_results}
        overall_result = ResultManagement().factory_method("new", ingredients)
        return overall_result


class ModelEvaluation:

    def __init__(self, config, model, evaluation: Union[Evaluation, None]):
        self.evaluation = evaluation
        self.config = config
        self.model = model
        self.model_eval_info = ModelEvaluationInformation(config, model)
        self.evaluation_results = {}

    def evaluate_timeseries_model(self, predictors, true_labels, stage: str, meta_data_training_set, meta_data_test_set, split_name: str =" split", ) -> None:

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

        result = EvalResultFactory.factory_method(optimizer_list, meta_data_training_set, meta_data_test_set, stage, evaluation_performed=True)

        self.evaluation_results[stage] = result

    def cross_validate_timeseries_model(self, data: TimeSeriesDataset) -> None:

        if self.evaluation.eval_info is None:
            print("Can not perform cross validation without evaluation information")
            return

        processing_meta_data = TimeSeriesMetaDataManagement.extract_procesing_meta_data(data.meta_data)
        labels = data.content["ards"]
        predictors = data.content.loc[:, data.content.columns != 'ards']
        n_splits = self.evaluation.eval_info.n_splits
        shuffle_cv = self.evaluation.eval_info.shuffle
        random_state = self.evaluation.eval_info.random_state
        # Create Splits for Crossvalidation
        cross_validation = StratifiedKFold(n_splits=n_splits,
                                           shuffle=shuffle_cv,
                                           random_state=random_state)

        if self.config["process"]["perform_threshold_optimization"]:
            threshold_optimizers = self.config['evaluation']['threshold_optimization_algorithms']
        else:
            threshold_optimizers = ["Standard"]

        optimizer_eval_dict = {}
        split_training_evaluation_dict = {}
        for optimizer in threshold_optimizers:
            optimizer_eval_dict[optimizer] = []
            split_training_evaluation_dict[optimizer] = []


        for (train_set, test_set), i in zip(cross_validation.split(predictors, labels),
                                            range(self.evaluation.eval_info.n_splits)):

            predictors_train = predictors.iloc[train_set]
            labels_train = labels.iloc[train_set]
            training_data_df = predictors_train.assign(ards=labels_train)

            training_storage_information = f"Training split {i} of data located at: {data.meta_data.additional_information}"

            training_data = TimeSeriesDatasetManagement.factory_method(training_data_df, processing_meta_data, training_storage_information, "CrossValidation")

            predictors_test = predictors.iloc[test_set]
            labels_test = labels.iloc[test_set]
            test_data_df = predictors_test.assign(ards=labels_test)

            test_storage_information = f"Test split {i} of data located at: {data.meta_data.additional_information}"
            test_data = TimeSeriesDatasetManagement.factory_method(test_data_df, processing_meta_data,
                                                                       test_storage_information, "CrossValidation")

            # Learn model for the split
            self.model.train_timeseries(training_data, self.config, "Training")
            training_eval = self.model.training_evaluation

            if self.config["process"]["save_models"]:
                if self.config["algorithm_base_path"][self.model.algorithm] != "default":
                    save_path = self.config["algorithm_base_path"][self.model.algorithm]
                else:
                    save_path = self.config["storage_path"] + "/" if self.config["storage_path"] else "./Save/" + str(
                        datetime.now().strftime("%m-%d-%Y_%H-%M-%S")) + "/"
                model_name =  self.model.name + "_split_" + str(i)
                old_model_name = self.model.name
                self.model.name = model_name
                self.model.save(save_path)
                self.model.name = old_model_name

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
                split_training_evaluation_dict[optimizer].append(training_result)
                optimizer_eval_dict[optimizer].append(eval_result)
        optimizer_eval_list = []
        optimizer_training_list = []
        for optimizer in threshold_optimizers:
            mean_split = SplitFactory.mean_split_factory_method(optimizer_eval_dict[optimizer])
            optimizer_eval_dict[optimizer].append(mean_split)
            complete_eval_list = optimizer_eval_dict[optimizer]
            optimizer_result = OptimizerFactory.factory_method(complete_eval_list, optimizer)
            optimizer_eval_list.append(optimizer_result)

            mean_training = SplitFactory.mean_split_factory_method(split_training_evaluation_dict[optimizer])
            split_training_evaluation_dict[optimizer].append(mean_training)
            complete_training_list = split_training_evaluation_dict[optimizer]
            optimizer_training_result = OptimizerFactory.factory_method(complete_training_list, optimizer)#
            optimizer_training_list.append(optimizer_training_result)


        eval_meta_data = data.meta_data
        eval_meta_data.additional_information = "To receive the training and test set generate the splits using the cross validation settings below."

        training_meta_data = data.meta_data
        training_meta_data.additional_information = "To receive the training and test set generate the splits using the cross validation settings below and use the training set as the test set. "

        eval_result_complete = EvalResultFactory.factory_method(optimizer_eval_list, eval_meta_data, eval_meta_data,"CrossValidation",
                                                                True, random_state,
                                                                shuffle_cv, n_splits)

        training_result_complete = EvalResultFactory.factory_method(optimizer_training_list, training_meta_data, training_meta_data,"CrossValidation",
                                                                True, random_state,
                                                                shuffle_cv, n_splits)


        self.evaluation_results["EvaluationCrossValidation"] = eval_result_complete
        self.evaluation_results["TrainingCrossValidation"] = training_result_complete

