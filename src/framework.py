from cli import make_parser
from processing import DataFileManager, DataProcessor, FeatureSelector, DataSegregator, TimeSeriesDatasetManagement, TimeSeriesDataset, DatasetGenerator
from ml_models import *
from evaluation import Evaluation
from metrics import ResultManagement

import os
import yaml
import json
import pandas as pd

from datetime import datetime
from pathlib import Path
import logging


logger = logging.getLogger(__name__)

class Framework:
    """
    Main framework class that orchestrates the entire ML pipeline including data loading,
    model training, evaluation, and cross-validation for timeseries and image data.
    """
    # ██╗███╗   ██╗██╗████████╗██╗ █████╗ ██╗     ██╗███████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
    # ██║████╗  ██║██║╚══██╔══╝██║██╔══██╗██║     ██║╚══███╔╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
    # ██║██╔██╗ ██║██║   ██║   ██║███████║██║     ██║  ███╔╝ ███████║   ██║   ██║██║   ██║██╔██╗ ██║
    # ██║██║╚██╗██║██║   ██║   ██║██╔══██║██║     ██║ ███╔╝  ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
    # ██║██║ ╚████║██║   ██║   ██║██║  ██║███████╗██║███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
    # ╚═╝╚═╝  ╚═══╝╚═╝   ╚═╝   ╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
    def __init__(self):
        """
        Initializes the Framework by loading configuration, setting up data/model managers,
        preparing output directory, and initializing model lists and processing objects.
        """
        logger.info("Initializing Framework...")
        
        args = make_parser().parse_args()
        if args.config:
            config = json.loads(args.config)
            logger.info("Configuration loaded from command line argument")
        else:
            with open(args.config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from file: {args.config_file}")

        self.config = config
        self.loader = DataFileManager()
        self.supported_timeseries_models = self.config['supported_algorithms']['timeseries_models']

        self.available_timeseries_models = {}

        for model in self.supported_timeseries_models:
            algorithm = model.replace("Model", "")
            self.available_timeseries_models[algorithm] = []

        self.timeseries_model_use_config = self.config['models']['timeseries_models']

        self.timeseries_models_to_train = self.create_needed_models(self.timeseries_model_use_config, 'to_train')
        self.timeseries_models_to_evaluate = self.create_needed_models(self.timeseries_model_use_config, 'to_evaluate')
        self.timeseries_models_to_execute = self.create_needed_models(self.timeseries_model_use_config, 'to_execute')
        self.timeseries_models_to_cross_validate = self.create_needed_models(self.timeseries_model_use_config, 'to_cross_validate')

        self.timeseries_training_set = None
        self.timeseries_test_set = None
        self.timeseries_complete_set = None

        self.timeseries_evaluations_result = None
        self.timeseries_cross_validation_result = None
        self.processing_meta_data = {}

        self.dataProcessor = DataProcessor(config["preprocessing"], config["data"]["database"], config["process"])
        self.feature_selector = FeatureSelector(config["feature_selection"])
        self.segregator = DataSegregator(config["data_segregation"])
        self.pneumonia_image_dataset = config["data"]["pneumonia_image_dataset"]
        self.ards_image_dataset = config["data"]["ards_image_dataset"]
        self.image_file_path = config["data"]["image_file_path"]
        self.method = config["image_model_parameters"]["method"]
        self.mode = config["image_model_parameters"]["mode"]
        self.image_model_use_config = self.config['models']['image_models']
        self.image_models_to_train = self.create_needed_models(self.image_model_use_config, 'to_train')
        self.image_models_to_evaluate = self.create_needed_models(self.image_model_use_config, 'to_evaluate')
        self.image_models_to_execute = self.create_needed_models(self.image_model_use_config, 'to_execute')
        self.image_models_to_cross_validate = self.create_needed_models(self.image_model_use_config, 'to_cross_validate')
        self.dataset_generator = DatasetGenerator()

        self.process = config["process"]
        self.model_base_paths = config["algorithm_base_path"]
        self.timeseries_file_path = config["data"]["timeseries_file_path"]
        self.timeseries_import_type = config["data"]["import_type"]

        self.outdir = config["storage_path"] if config["storage_path"] else "./Save/" + str(
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + "/"
        Path(self.outdir).mkdir(parents=True, exist_ok=True)
        if not self.outdir.endswith("/"):
            self.outdir += "/"

        logger.info(f"Output directory set to: {self.outdir}")

        self.TimeseriesModelManager = TimeSeriesModelManager(config, self.outdir)
        logger.info("Framework initialization completed successfully")

    @staticmethod
    def create_needed_models(config: dict, stage: str) -> dict[str, list[str]]:
        """
        Creates a dictionary of models needed for a specific stage (to_train, to_evaluate, etc.)
        by filtering the config for active models only.
        
        Args:
            config: Configuration dictionary containing model settings
            stage: The pipeline stage (to_train, to_evaluate, to_execute, to_cross_validate)
            
        Returns:
            Dictionary with active models for the specified stage
        """
        stage_config = config[stage]
        result_dict = {}
        for model_algorithm in stage_config.keys():
            if stage_config[model_algorithm]["Active"]:
                content = stage_config[model_algorithm]
                del content["Active"]
                result_dict[model_algorithm] = content
        return result_dict

    # ████████╗██╗███╗   ███╗███████╗    ███████╗███████╗██████╗ ██╗███████╗███████╗    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗     ███████╗
    # ╚══██╔══╝██║████╗ ████║██╔════╝    ██╔════╝██╔════╝██╔══██╗██║██╔════╝██╔════╝    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     ██╔════╝
    #    ██║   ██║██╔████╔██║█████╗█████╗███████╗█████╗  ██████╔╝██║█████╗  ███████╗    ██╔████╔██║██║   ██║██║  ██║█████╗  ██║     ███████╗
    #    ██║   ██║██║╚██╔╝██║██╔══╝╚════╝╚════██║██╔══╝  ██╔══██╗██║██╔══╝  ╚════██║    ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     ╚════██║
    #    ██║   ██║██║ ╚═╝ ██║███████╗    ███████║███████╗██║  ██║██║███████╗███████║    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗███████║
    #    ╚═╝   ╚═╝╚═╝     ╚═╝╚══════╝    ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚══════╝
    def load_timeseries_data(self):
        """
        Loads timeseries data from file, applies preprocessing and feature selection,
        segregates data into training and test sets, and saves all datasets with metadata.
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Loading and Processing Timeseries Data")
        logger.info("=" * 80)
        
        logger.info(f"Loading timeseries data from: {self.timeseries_file_path} with type: {self.timeseries_import_type}")
        dataframe, dataset_metadata = self.loader.load_file(self.timeseries_file_path, self.timeseries_import_type)
        logger.info(f"Successfully loaded data with shape: {dataframe.shape}")

        if dataset_metadata:
            self.dataProcessor.database_name = dataset_metadata.datasource
            logger.info(f"Dataset source: {dataset_metadata.datasource}")

        dataframe = self.dataProcessor.process_data(dataframe, dataset_metadata)
        processing_meta_data = self.dataProcessor.processing_meta_data()
        logger.info(f"Data preprocessing completed. Shape after preprocessing: {dataframe.shape}")

        if self.process["perform_feature_selection"]:
            logger.info("Performing feature selection...")
            dataframe = self.feature_selector.perform_feature_selection(dataframe)
            self.feature_selector.create_meta_data()
            logger.info(f"Feature selection completed. Number of features: {dataframe.shape[1]}")
        processing_meta_data["feature_selection"] = self.feature_selector.meta_data

        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)

        logger.info("Finished preprocessing and saved result to file!")
        
        if self.process["perform_data_segregation"]:
            logger.info("Performing data segregation into training and test sets...")
            training_data, test_data = self.segregator.segregate_data(dataframe)
            logger.info(f"Training set shape: {training_data.shape}, Test set shape: {test_data.shape}")
        else:
            logger.warning("Warning: Training and Test data are the same!")
            training_data = test_data = dataframe

        current_time_str = str(datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
        name = self.config["evaluation"]["evaluation_name"].replace(" ", "_")
        base_path_data = self.outdir + name + "_data"
        path_complete = base_path_data + "_complete"
        path_training = base_path_data + "_training"
        path_test = base_path_data + "_test"

        self.timeseries_training_set = TimeSeriesDatasetManagement.factory_method(training_data, processing_meta_data, path_training, "Training")
        self.timeseries_test_set = TimeSeriesDatasetManagement.factory_method(test_data, processing_meta_data, path_test, "Test")
        self.timeseries_complete_set = TimeSeriesDatasetManagement.factory_method(dataframe, processing_meta_data, path_complete, "Complete")
        
        TimeSeriesDatasetManagement.write(self.timeseries_complete_set)
        TimeSeriesDatasetManagement.write(self.timeseries_training_set)
        TimeSeriesDatasetManagement.write(self.timeseries_test_set)
        logger.info("All datasets written successfully")

    def load_timeseries_models(self, stage: str):
        """
        Loads pre-trained timeseries models for a specific stage (to_execute or to_evaluate).
        
        Args:
            stage: The pipeline stage (to_execute, to_evaluate, etc.)
        """
        logger.info(f"Loading timeseries models for stage: {stage}")
        
        if stage == "to_execute":
            needed_models = self.timeseries_models_to_execute
        elif stage == "to_evaluate":
            needed_models = self.timeseries_models_to_evaluate
        else:
            logger.info(f"For stage '{stage}' it does not make sense to load models")
            return
            
        self.available_timeseries_models = self.TimeseriesModelManager.load_models(needed_models, self.available_timeseries_models, self.model_base_paths)
        logger.info(f"Models loaded successfully for stage: {stage}")

    def learn_timeseries_models(self):
        """
        Trains all configured timeseries models on the training dataset.
        Saves trained models if configured and stores them in the available models list.
        """
        logger.info("=" * 80)
        logger.info("STEP 2: Training Timeseries Models")
        logger.info("=" * 80)

        model_dict = self.TimeseriesModelManager.create_model_from_config(self.timeseries_models_to_train,
                                                                          self.timeseries_model_use_config["base_path_config"]["to_train"])

        total_models = sum(len(models) for models in model_dict.values())
        current_model = 0

        for model_type, models in model_dict.items():
            for model in models:
                current_model += 1
                logger.info(f"[{current_model}/{total_models}] Starting training of '{model.name}' (Algorithm: {model.algorithm})")
                model.train_timeseries(self.timeseries_training_set, self.config, "Training")
                
                if self.process["save_models"]:
                    if self.config["algorithm_base_path"][model.algorithm] != "default":
                        save_path = self.config["algorithm_base_path"][model.algorithm]
                    else:
                        save_path = self.outdir
                    model.save(save_path)
                    logger.info(f"Model '{model.name}' saved to: {save_path}")
                    
                logger.info(f"[{current_model}/{total_models}] Successfully trained '{model.name}' for algorithm '{model.algorithm}'")
                self.available_timeseries_models[model.algorithm].append(model)

        logger.info(f"Model training completed. Total models trained: {total_models}")

    def execute_timeseries_models(self, test_set: TimeSeriesDataset):
        """
        Executes (predicts) trained timeseries models on the test set and saves predictions to CSV files.
        
        Args:
            test_set: TimeSeriesDataset object containing test data
        """
        logger.info("=" * 80)
        logger.info("STEP 3: Executing Timeseries Models (Making Predictions)")
        logger.info("=" * 80)
        
        self.load_timeseries_models("to_execute")
        test_data = test_set.content
        input_features = test_data.drop(columns=['ards'])
        test_data = test_data.rename(columns={"ards": "ards_diagnosed"}).reset_index(drop=True)
        logger.info(f"Test set prepared with {len(input_features)} samples and {len(input_features.columns)} features")

        total_predictions = 0
        for model_type, model_names in self.timeseries_models_to_execute.items():
            for model in self.available_timeseries_models[model_type]:
                if model.name not in model_names["Names"]:
                    continue
                    
                logger.info(f"Making predictions with model '{model.name}' ({model.algorithm})...")
                prediction = model.predict(input_features)
                logger.info(f"Finished prediction of '{model.name}'. Generated {len(prediction)} predictions")
                
                df = pd.DataFrame({"ards_predicted": prediction}).reset_index(drop=True)
                df = pd.concat([test_data, df], axis=1)
                name = self.config["evaluation"]["evaluation_name"].replace(" ", "_")
                output_file = self.outdir + f"{name}_prediction_{model.algorithm}_{model.name}.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Predictions saved to: {output_file}")
                total_predictions += 1

        logger.info(f"Execution completed. Total predictions generated: {total_predictions}")

    def evaluate_timeseries_models(self):
        """
        Evaluates trained timeseries models on the test set and computes evaluation metrics.
        """
        logger.info("=" * 80)
        logger.info("STEP 4: Evaluating Timeseries Models")
        logger.info("=" * 80)
        
        self.load_timeseries_models("to_evaluate")

        models_to_evaluate_dict = {}
        for model_type, model_name in self.timeseries_models_to_evaluate.items():
            for model in self.available_timeseries_models[model_type]:
                if model.name not in model_name["Names"]:
                    continue
                if model_type in models_to_evaluate_dict:
                    models_to_evaluate_dict[model_type].append(model)
                else:
                    models_to_evaluate_dict[model_type] = [model]

        total_eval_models = sum(len(models) for models in models_to_evaluate_dict.values())
        logger.info(f"Starting evaluation of {total_eval_models} models...")

        evaluator = Evaluation(self.config, dataset_training=self.timeseries_training_set,
                               dataset_test=self.timeseries_test_set)
        overall_result = evaluator.evaluate_timeseries_models(models_to_evaluate_dict)
        self.timeseries_evaluations_result = overall_result
        logger.info(f"Evaluation completed successfully for {total_eval_models} models")

    def cross_validate_models(self):
        """
        Performs k-fold cross-validation on timeseries models using the training dataset.
        """
        logger.info("=" * 80)
        logger.info("STEP 5: Performing Cross-Validation")
        logger.info("=" * 80)
        
        evaluator = Evaluation(self.config, dataset_training=self.timeseries_training_set,
                               dataset_test=self.timeseries_test_set)

        logger.info("Creating models for cross-validation...")
        models_to_cross_validate_dict = self.TimeseriesModelManager.create_model_from_config(
            self.timeseries_models_to_cross_validate, self.timeseries_model_use_config["base_path_config"]["to_cross_validate"])

        total_cv_models = sum(len(models) for models in models_to_cross_validate_dict.values())
        if total_cv_models == 0:
            logger.warning("No models configured for cross-validation. Skipping this step.")
            return
        else:
            logger.info(f"Starting cross-validation for {total_cv_models} models...")
            overall_result = evaluator.cross_validate_timeseries_models(models_to_cross_validate_dict)
            self.timeseries_cross_validation_result = overall_result
            logger.info(f"Cross-validation completed successfully for {total_cv_models} models")

    def handle_timeseries_results(self):
        """
        Processes and saves evaluation results. Merges cross-validation and evaluation results
        if both are available, then saves to JSON file.
        """
        logger.info("=" * 80)
        logger.info("STEP 6: Handling and Saving Results")
        logger.info("=" * 80)

        eval_name = self.config['evaluation']['evaluation_name']
        eval_name = eval_name.replace(" ", "_")
        result_location = self.outdir + f'{eval_name}_results.json'

        if self.timeseries_cross_validation_result and self.timeseries_evaluations_result:
            logger.info("Merging cross-validation and evaluation results...")
            cv_result = self.timeseries_cross_validation_result
            eval_result = self.timeseries_evaluations_result

            content = {"eval_name": eval_name, "storage_location": result_location, "CV": cv_result, "Eval": eval_result}
            final_result = ResultManagement().merge(content, "CV_EVAL")
            logger.info("Results merged successfully")

        elif self.timeseries_cross_validation_result:
            logger.info("Using cross-validation results only...")
            final_result = self.timeseries_cross_validation_result
            eval_name = self.config['evaluation']['evaluation_name']

        elif self.timeseries_evaluations_result:
            logger.info("Using evaluation results only...")
            final_result = self.timeseries_evaluations_result

        else:
            logger.error("This should never happen - no results available to save")
            return

        logger.info(f"Saving results to: {result_location}")
        with open(result_location, 'w', encoding='utf-8') as f:
            f.write(final_result.model_dump_json(indent=4))
        logger.info("Results saved successfully")

    # ██╗███╗   ███╗ █████╗  ██████╗ ███████╗    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗     ███████╗
    # ██║████╗ ████║██╔══██╗██╔════╝ ██╔════╝    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     ██╔════╝
    # ██║██╔████╔██║███████║██║  ███╗█████╗      ██╔████╔██║██║   ██║██║  ██║█████╗  ██║     ███████╗
    # ██║██║╚██╔╝██║██╔══██║██║   ██║██╔══╝      ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     ╚════██║
    # ██║██║ ╚═╝ ██║██║  ██║╚██████╔╝███████╗    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗███████║
    # ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚══════╝
    def load_image_data(self):
        """
        Loads timeseries data from file, applies preprocessing and feature selection,
        segregates data into training and test sets, and saves all datasets with metadata.
        """
        logger.info("=" * 80)
        logger.info("STEP 6: Loading and Processing Image Data")
        logger.info("=" * 80)

        for dl_method in self.image_dl_methods:
            self.image_pneumonia_training_data = self.dataset_generator.build_dataset(self.pneumonia_dataset, dl_method,
                                                                                      'PNEUMONIA',
                                                                                      path=self.image_file_path,
                                                                                      augment=False)
            self.image_ards_training_data = self.dataset_generator.build_dataset(self.ards_dataset, dl_method, 'ARDS',
                                                                                 path=self.image_file_path,
                                                                                 augment=False)
            self.image_ards_test_data = self.dataset_generator.build_dataset('test', dl_method, 'ARDS',
                                                                             path=self.image_file_path, augment=False)


    def load_image_models(self):
        """
        Loads pre-trained image classification models.
        Currently not implemented.
        """
        logger.warning("load_image_models: Feature not yet implemented")
        raise NotImplementedError

    def learn_image_models(self):
        """
        Trains image classification models.
        Currently not implemented.
        """
        logger.warning("learn_image_models: Feature not yet implemented")
        raise NotImplementedError

    def execute_image_models(self):
        """
        Executes image classification models.
        Currently not implemented.
        """
        logger.warning("execute_image_models: Feature not yet implemented")
        raise NotImplementedError

    def evaluate_image_models(self):
        """
        Evaluates image classification models.
        Currently not implemented.
        """
        logger.warning("evaluate_image_models: Feature not yet implemented")
        raise NotImplementedError

    # ███████╗██████╗  █████╗ ███╗   ███╗███████╗██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗    ██████╗ ██╗   ██╗███╗   ██╗
    # ██╔════╝██╔══██╗██╔══██╗████╗ ████║██╔════╝██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝    ██╔══██╗██║   ██║████╗  ██║
    # █████╗  ██████╔╝███████║██╔████╔██║█████╗  ██║ █╗ ██║██║   ██║██████╔╝█████╔╝     ██████╔╝██║   ██║██╔██╗ ██║
    # ██╔══╝  ██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝  ██║███╗██║██║   ██║██╔══██╗██╔═██╗     ██╔══██╗██║   ██║██║╚██╗██║
    # ██║     ██║  ██║██║  ██║██║ ╚═╝ ██║███████╗╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗    ██║  ██║╚██████╔╝██║ ╚████║
    # ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝    ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
    def run(self):
        """
        Main entry point that orchestrates the entire ML pipeline.
        Executes all configured steps in sequence: data loading, model training,
        predictions, evaluation, cross-validation, and result handling.
        """
        logger.info("=" * 80)
        logger.info("STARTING MACHINE LEARNING FRAMEWORK")
        logger.info("=" * 80)
        
        # Store configuration in outdir
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
        
        logger.info(f"Saving configuration to: {self.outdir}config.json")
        with open(self.outdir + 'config.json', 'w') as f:
            json.dump(self.config, f)

        if self.process["load_timeseries_data"]:
            self.load_timeseries_data()
        else:
            logger.info("Skipping STEP 1: load_timeseries_data (disabled in config)")

        if self.process["perform_timeseries_training"]:
            if not self.timeseries_training_set:
                logger.error("Cannot train without training data. Exiting...")
                exit()
            self.learn_timeseries_models()
        else:
            logger.info("Skipping STEP 2: perform_timeseries_training (disabled in config)")

        if self.process["perform_timeseries_classification"]:
            if not self.timeseries_test_set:
                logger.error("Cannot predict without test data. Exiting...")
                exit()
            self.execute_timeseries_models(self.timeseries_test_set)
        else:
            logger.info("Skipping STEP 3: perform_timeseries_classification (disabled in config)")

        if self.process["calculate_evaluation_metrics"]:
            self.evaluate_timeseries_models()
        else:
            logger.info("Skipping STEP 4: calculate_evaluation_metrics (disabled in config)")

        if self.process["perform_cross_validation"]:
            if not self.timeseries_training_set:
                logger.error("Cannot cross validate without training data. Exiting...")
                exit()
            self.cross_validate_models()
        else:
            logger.info("Skipping STEP 5: perform_cross_validation (disabled in config)")

        if self.timeseries_evaluations_result or self.timeseries_cross_validation_result:
            self.handle_timeseries_results()

        if self.process["load_image_data"]:
            logger.info("Currently not supported: load_image_data")
            self.load_image_data()
        else:
            logger.info("Skipping STEP 6: load_image_data (disabled in config)")

        if self.process["train_image_models"]:
            logger.info("Currently not supported: train_image_models")
            self.learn_image_models()
        else:
            logger.info("Skipping STEP 7: train_image_data (disabled in config)")

        if self.process["execute_image_models"]:
            logger.info("Currently not supported: execute_image_models")
            self.execute_image_models()
        else:
            logger.info("Skipping STEP 8: execute_image_models (disabled in config)")

        if self.process["test_image_models"]:
            logger.info("Currently not supported: test_image_models")
            self.evaluate_image_models()
        else:
            logger.info("Skipping STEP 9: evaluate_image_models (disabled in config)")

        logger.info("=" * 80)
        logger.info("MACHINE LEARNING FRAMEWORK EXECUTION COMPLETED")
        logger.info("=" * 80)
