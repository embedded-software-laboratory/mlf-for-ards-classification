import copy
import logging
from typing import Union
from datetime import datetime

from evaluation.EvaluationInformation import EvaluationInformation, ModelEvaluationInformation

from processing import TimeSeriesDataset, TimeSeriesDatasetManagement, TimeSeriesMetaDataManagement
from ml_models import Model
from metrics.ModelFactories import ExperimentResult, ModelResultFactory, ResultManagement, SplitFactory, OptimizerFactory, EvalResultFactory

from sklearn.model_selection import StratifiedKFold


logger = logging.getLogger(__name__)


class Evaluation:
    """
    Coordinates evaluation workflows:
    - Evaluate pre-trained models on a held-out test set
    - Cross-validate models on a training dataset

    This class collects results and wraps them into the ExperimentResult structure
    used by the rest of the framework.
    """

    def __init__(self, config, dataset_training: TimeSeriesDataset, dataset_test: TimeSeriesDataset):
        logger.info("Initializing Evaluation manager...")
        self.config = config
        self.eval_info = EvaluationInformation(config, dataset_training, dataset_test)
        self.model_results = {}
        logger.info("Evaluation manager initialized.")

    def evaluate_timeseries_models(self, models_to_evaluate_dict: dict[str, list[Model]]) -> ExperimentResult:
        """
        Evaluates a set of timeseries models on the test dataset provided in EvaluationInformation.
        Skips untrained models and stores per-model results in self.model_results.
        Returns an ExperimentResult containing all model evaluations.
        """
        logger.info("Starting evaluation of timeseries models...")
        for model_algorithm in models_to_evaluate_dict:
            logger.info(f"Evaluating models for algorithm {model_algorithm}")
            for timeseries_model in models_to_evaluate_dict[model_algorithm]:
                logger.info(f"Evaluating model: {timeseries_model.name}")
                if not timeseries_model.trained:
                    logger.info(f"Skipping untrained model: {timeseries_model.name}")
                    continue

                model_evaluation = ModelEvaluation(self.config, timeseries_model, self)
                predictors = self.eval_info.dataset_test.content.loc[:, self.eval_info.dataset_test.content.columns != 'ards']
                meta_data_training_set = self.eval_info.dataset_training.meta_data
                meta_data_test_set = self.eval_info.dataset_test.meta_data
                labels = self.eval_info.dataset_test.content['ards']

                logger.debug(f"Predictors shape: {predictors.shape}, Labels shape: {labels.shape}")
                model_evaluation.evaluate_timeseries_model(predictors, labels, "Evaluation", meta_data_training_set, meta_data_test_set)

                model_result = ModelResultFactory.factory_method(
                    model_evaluation.model_eval_info,
                    model_evaluation.evaluation_results,
                    model_evaluation.model.training_evaluation,
                    "Evaluation"
                )
                self.model_results[timeseries_model.name] = model_result
                logger.info(f"Stored evaluation result for model: {timeseries_model.name}")

        ingredients = {"EvaluationInformation": self.eval_info, "model_results": self.model_results}
        overall_result = ResultManagement().factory_method("new", ingredients)
        logger.info("Model evaluation completed.")
        return overall_result

    def cross_validate_timeseries_models(self, models_to_cross_validate_dict: dict[str, list[Model]]) -> ExperimentResult:
        """
        Performs cross-validation for each model listed in models_to_cross_validate_dict.
        Returns an ExperimentResult aggregating cross-validation outcomes.
        """
        logger.info("Starting cross-validation for timeseries models...")
        for model_algorithm in models_to_cross_validate_dict:
            for timeseries_model in models_to_cross_validate_dict[model_algorithm]:
                logger.info(f"Cross-validating model: {timeseries_model.name}")
                model_evaluation = ModelEvaluation(self.config, timeseries_model, self)
                model_evaluation.cross_validate_timeseries_model(self.eval_info.dataset_training)
                model_result = ModelResultFactory.factory_method(
                    model_evaluation.model_eval_info,
                    model_evaluation.evaluation_results,
                    model_evaluation.model.training_evaluation,
                    stage="CrossValidation"
                )
                self.model_results[timeseries_model.name] = model_result
                logger.info(f"Stored cross-validation result for model: {timeseries_model.name}")

        ingredients = {"EvaluationInformation": self.eval_info, "model_results": self.model_results}
        overall_result = ResultManagement().factory_method("new", ingredients)
        return overall_result

    def evaluate_image_models(self, models_to_evaluate_dict: dict[str, list[Model]], 
                             pneumonia_dataset: str, ards_dataset: str, method: str, mode: str) -> ExperimentResult:
        """
        Evaluates a set of image models by reading their pre-computed test metrics from CSV files.
        Image models save their metrics during test_image_model(), so this method reads those
        results and packages them into the standard ExperimentResult format.
        
        Args:
            models_to_evaluate_dict: Dictionary mapping model types to lists of models
            pneumonia_dataset: Name of the pneumonia dataset used
            ards_dataset: Name of the ARDS dataset used
            method: Method name used for training
            mode: Mode used for training
            
        Returns:
            ExperimentResult containing all model evaluations
        """
        import csv
        import os
        import re
        
        def parse_metric_value(value_str):
            """
            Parse metric value that can be in different formats:
            - 'tensor(0.8614)' -> 0.8614
            - 'tensor(0.8614, device='cuda:0')' -> 0.8614
            - '[0.8614]' -> 0.8614
            - '0.8614' -> 0.8614
            """
            if not isinstance(value_str, str):
                return float(value_str)
            
            # Remove whitespace
            value_str = value_str.strip()
            
            # Handle tensor format: tensor(0.8614) or tensor(0.8614, device='cuda:0')
            tensor_match = re.search(r'tensor\(([-+]?[0-9]*\.?[0-9]+)', value_str)
            if tensor_match:
                return float(tensor_match.group(1))
            
            # Handle list format: [0.8614]
            if value_str.startswith('[') and value_str.endswith(']'):
                value_str = value_str.strip('[]').strip()
            
            # Handle regular float
            return float(value_str)
        
        logger.info("Starting evaluation of image models...")
        for model_algorithm in models_to_evaluate_dict:
            logger.info(f"Evaluating models for algorithm {model_algorithm}")
            for image_model in models_to_evaluate_dict[model_algorithm]:
                logger.info(f"Evaluating model: {image_model.name}")
                
                # Construct the path to the saved metrics file
                dataset_name = f"{pneumonia_dataset}_{ards_dataset}"
                metrics_filename = f'test_metrics_{image_model.name}_{dataset_name}_{method}_{mode}.pt'
                metrics_path = os.path.join(image_model.path_results_ards, metrics_filename)
                
                if not os.path.exists(metrics_path):
                    logger.warning(f"Metrics file not found for model '{image_model.name}': {metrics_path}")
                    logger.warning(f"Skipping evaluation for this model")
                    continue
                
                # Read the CSV file with test metrics
                try:
                    with open(metrics_path, 'r') as f:
                        reader = csv.DictReader(f)
                        test_results = next(reader)  # Read first (and only) row
                    
                    # Extract metrics - handle tensor strings, lists, or plain floats
                    test_metrics = {
                        'test_loss': parse_metric_value(test_results['test_loss']),
                        'test_accuracy': parse_metric_value(test_results['test_acc']),
                        'test_precision': parse_metric_value(test_results['test_prec']),
                        'test_recall': parse_metric_value(test_results['test_recall']),
                        'test_specificity': parse_metric_value(test_results['test_specificity']),
                        'test_auroc': parse_metric_value(test_results['test_auroc']),
                        'test_f1': parse_metric_value(test_results['test_f1'])
                    }
                    
                    logger.info(f"Successfully loaded metrics for '{image_model.name}':")
                    logger.info(f"  Accuracy: {test_metrics['test_accuracy']:.4f}")
                    logger.info(f"  AUROC: {test_metrics['test_auroc']:.4f}")
                    logger.info(f"  F1: {test_metrics['test_f1']:.4f}")
                    
                    # Create ModelEvaluationInformation for image model
                    model_eval_info = ModelEvaluationInformation(self.config, image_model)
                    
                    # Create evaluation results structure
                    # For image models, we have pre-computed metrics, so we create a simplified structure
                    evaluation_results = {
                        'Evaluation': {
                            'Standard': {
                                'metrics': test_metrics,
                                'threshold': 0.5  # Standard threshold for binary classification
                            }
                        }
                    }
                    
                    # Create ModelResult using factory method
                    model_result = ModelResultFactory.factory_method(
                        model_eval_info,
                        evaluation_results,
                        training_evaluation=None,  # Image models don't store training evaluation the same way
                        stage="Evaluation"
                    )
                    
                    self.model_results[image_model.name] = model_result
                    logger.info(f"Stored evaluation result for model: {image_model.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to read or process metrics for '{image_model.name}': {e}")
                    logger.warning(f"Skipping evaluation for this model")
                    continue
        
        ingredients = {"EvaluationInformation": self.eval_info, "model_results": self.model_results}
        overall_result = ResultManagement().factory_method("new", ingredients)
        logger.info("Image model evaluation completed.")
        return overall_result


class ModelEvaluation:
    """
    Encapsulates evaluation logic for a single model:
    - Running a single evaluation on a test set
    - Performing cross-validation and aggregating split results
    """

    def __init__(self, config, model, evaluation: Union[Evaluation, None]):
        self.evaluation = evaluation
        self.config = config
        self.model = model
        self.model_eval_info = ModelEvaluationInformation(config, model)
        self.evaluation_results = {}
        logger.debug("ModelEvaluation initialized.")

    def evaluate_timeseries_model(self, predictors, true_labels, stage: str, meta_data_training_set, meta_data_test_set, split_name: str =" split", ) -> None:
        """
        Evaluate the model on provided predictors/labels and produce EvalResult objects.
        Handles both probability-output and label-output models and optionally applies
        threshold optimization algorithms defined in config.
        """
        logger.info(f"Evaluating model '{self.model.name}' on stage: {stage}")
        eval_split_name = stage + split_name

        # drop patient_id column injected into predictors if present
        if "patient_id" in predictors.columns:
            eval_predictors = predictors.drop("patient_id", axis=1)
            logger.debug("Dropped 'patient_id' from predictors for evaluation.")
        else:
            eval_predictors = predictors

        # Obtain model outputs
        if self.model.has_predict_proba():
            logger.debug("Model supports predict_proba; computing predicted probabilities.")
            self.model_eval_info.predicted_probas_test = self.model.predict_proba(eval_predictors)[:, 1]
        else:
            logger.debug("Model does not support predict_proba; computing predicted labels.")
            self.model_eval_info.predicted_labels_test = self.model.predict(eval_predictors)

        self.model_eval_info.true_labels_test = true_labels

        # Determine threshold optimizers
        if self.config["process"]["perform_threshold_optimization"]:
            threshold_optimizers = self.config['evaluation']['threshold_optimization_algorithms']
            logger.debug(f"Using threshold optimizers from config: {threshold_optimizers}")
        else:
            threshold_optimizers = ["Standard"]
            logger.debug("Threshold optimization disabled; using 'Standard' optimizer.")

        optimizer_list = []
        for optimizer in threshold_optimizers:
            eval_result = SplitFactory.factory_method(self.model_eval_info, eval_split_name, optimizer, stage)
            optimizer_result = OptimizerFactory.factory_method([eval_result], optimizer)
            optimizer_list.append(optimizer_result)
            logger.debug(f"Created optimizer result for optimizer: {optimizer}")

        result = EvalResultFactory.factory_method(optimizer_list, meta_data_training_set, meta_data_test_set, stage, evaluation_performed=True)
        self.evaluation_results[stage] = result
        logger.info(f"Evaluation for model '{self.model.name}' on stage '{stage}' completed and stored.")

    def cross_validate_timeseries_model(self, data: TimeSeriesDataset) -> None:
        """
        Perform stratified K-fold cross-validation on provided TimeSeriesDataset.
        Trains the model for each split, optionally saves split-specific models and
        aggregates per-split optimizer/evaluation results.
        """
        logger.info(f"Starting cross-validation for model: {self.model.name}")
        if self.evaluation.eval_info is None:
            logger.warning("Evaluation information missing - cannot perform cross validation.")
            return

        processing_meta_data = TimeSeriesMetaDataManagement.extract_procesing_meta_data(data.meta_data)
        labels = data.content["ards"]
        predictors = data.content.loc[:, data.content.columns != 'ards']
        n_splits = self.evaluation.eval_info.n_splits
        shuffle_cv = self.evaluation.eval_info.shuffle
        random_state = self.evaluation.eval_info.random_state

        logger.debug(f"Cross-validation settings - n_splits: {n_splits}, shuffle: {shuffle_cv}, random_state: {random_state}")

        cross_validation = StratifiedKFold(n_splits=n_splits, shuffle=shuffle_cv, random_state=random_state)

        if self.config["process"]["perform_threshold_optimization"]:
            threshold_optimizers = self.config['evaluation']['threshold_optimization_algorithms']
            logger.debug(f"Threshold optimizers: {threshold_optimizers}")
        else:
            threshold_optimizers = ["Standard"]

        optimizer_eval_dict = {opt: [] for opt in threshold_optimizers}
        split_training_evaluation_dict = {opt: [] for opt in threshold_optimizers}

        # Iterate over CV splits
        for (train_set, test_set), i in zip(cross_validation.split(predictors, labels), range(self.evaluation.eval_info.n_splits)):
            logger.info(f"Processing CV split {i+1}/{n_splits}")

            predictors_train = predictors.iloc[train_set]
            labels_train = labels.iloc[train_set]
            training_data_df = predictors_train.assign(ards=labels_train)

            training_storage_information = f"Training split {i} of data located at: {data.meta_data.additional_information}"
            training_data = TimeSeriesDatasetManagement.factory_method(training_data_df, processing_meta_data, training_storage_information, "CrossValidation")

            predictors_test = predictors.iloc[test_set]
            labels_test = labels.iloc[test_set]
            test_data_df = predictors_test.assign(ards=labels_test)

            test_storage_information = f"Test split {i} of data located at: {data.meta_data.additional_information}"
            test_data = TimeSeriesDatasetManagement.factory_method(test_data_df, processing_meta_data, test_storage_information, "CrossValidation")

            # Train model for the split
            logger.debug("Training model for current split...")
            self.model.train_timeseries(training_data, self.config, "Training")
            training_eval = self.model.training_evaluation

            # Optionally save split model
            if self.config["process"]["save_models"]:
                if self.config["algorithm_base_path"][self.model.algorithm] != "default":
                    save_path = self.config["algorithm_base_path"][self.model.algorithm]
                else:
                    save_path = self.config["storage_path"] + "/" if self.config["storage_path"] else "./Save/" + str(datetime.now().strftime("%m-%d-%Y_%H-%M-%S")) + "/"
                model_name =  self.model.name + "_split_" + str(i)
                old_model_name = self.model.name
                self.model.name = model_name
                logger.debug(f"Saving model for split {i} to {save_path}")
                self.model.save(save_path)
                self.model.name = old_model_name

            # Prepare evaluation predictors (drop patient_id if present)
            if "patient_id" in predictors_test.columns:
                eval_predictors_test = predictors_test.drop("patient_id", axis=1)
            else:
                eval_predictors_test = predictors_test
            if "patient_id" in predictors_train.columns:
                eval_predictors_train = predictors_train.drop("patient_id", axis=1)
            else:
                eval_predictors_train = predictors_train

            # Obtain model predictions for train and test splits
            if self.model.has_predict_proba():
                self.model_eval_info.predicted_probas_test = self.model.predict_proba(eval_predictors_test)[:, 1]
                self.model_eval_info.predicted_probas_training = self.model.predict_proba(eval_predictors_train)[:, 1]
            else:
                self.model_eval_info.predicted_labels_test = self.model.predict(eval_predictors_test)
                self.model_eval_info.predicted_labels_training = self.model.predict(eval_predictors_train)

            self.model_eval_info.true_labels_test = labels_test
            self.model_eval_info.true_labels_training = labels_train

            # Collect per-optimizer split results
            for optimizer in threshold_optimizers:
                training_result = copy.deepcopy(training_eval.contained_optimizers[optimizer].contained_splits["Training split"])
                eval_result = SplitFactory.factory_method(self.model_eval_info, f"CrossValidationEvaluation split {i}", optimizer, "Evaluation")
                training_result.split_name = f"CrossValidationTraining split: {i}"
                split_training_evaluation_dict[optimizer].append(training_result)
                optimizer_eval_dict[optimizer].append(eval_result)
                logger.debug(f"Appended split results for optimizer: {optimizer}, split: {i}")

        # Aggregate results across splits and produce final EvalResult objects
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
            optimizer_training_result = OptimizerFactory.factory_method(complete_training_list, optimizer)
            optimizer_training_list.append(optimizer_training_result)

            logger.debug(f"Aggregated optimizer results for: {optimizer}")

        # Prepare metadata for EvalResultFactory
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
        logger.info(f"Cross-validation finished for model: {self.model.name}")

