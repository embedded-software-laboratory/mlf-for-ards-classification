import yaml
import logging

from ml_models.timeseries_model import TimeSeriesModel
from ml_models.adaboost import AdaBoostModel
from ml_models.lightGBM import LightGBMModel
from ml_models.logistic_regression import LogisticRegressionModel
from ml_models.random_forest import RandomForestModel
#from ml_models.recurrentneuralnetworkmodel import RecurrentNeuralNetworkModel
from ml_models.support_vector_machine import SupportVectorMachineModel
from ml_models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)


class TimeSeriesModelManager:
    """
    Manages the creation, configuration, and loading of timeseries machine learning models.
    Handles model instantiation from configuration files and manages model persistence.
    """

    def __init__(self, config, path):
        """
        Initializes the TimeSeriesModelManager with configuration and output directory.
        
        Args:
            config: Global framework configuration dictionary
            path: Output directory path where models will be saved
        """
        logger.info("Initializing TimeSeriesModelManager...")
        self.config = config
        self.save_models = config["process"]["save_models"]
        self.outdir = path
        logger.info(f"Save models enabled: {self.save_models}")
        logger.info(f"Output directory: {self.outdir}")
        logger.info("TimeSeriesModelManager initialized successfully")

    def create_model_from_config(self, needed_models: dict, base_config_path: str ):
        """
        Creates model instances from configuration specifications.
        Instantiates models with configured names and loads custom hyperparameters if provided.
        
        Args:
            needed_models: Dictionary with structure:
                {
                    'ModelType': {
                        'Names': ['model_name1', 'model_name2', ...],
                        'Configs': ['config_file1.yml', 'config_file2.yml', ...]
                    }
                }
            base_config_path: Base path to hyperparameter configuration files
            
        Returns:
            Dictionary mapping model types to lists of configured model instances
        """
        logger.info("=" * 80)
        logger.info("Creating Models from Configuration")
        logger.info("=" * 80)
        
        models = {}
        total_models = sum(len(needed_models[model_type]["Names"]) for model_type in needed_models)
        current_model_count = 0
        
        for model_type in needed_models:
            names = needed_models[model_type]["Names"]
            configs = needed_models[model_type]["Configs"]
            logger.info(f"Creating {len(names)} models of type: {model_type}")
            
            for i in range(len(names)):
                current_model_count += 1
                logger.info(f"[{current_model_count}/{total_models}] Creating model '{names[i]}' ({model_type})...")
                
                # Instantiate the model based on type
                try:
                    model = eval(model_type + "Model()")
                    logger.debug(f"Model instance created for {model_type}")
                except Exception as e:
                    logger.error(f"Failed to instantiate model {model_type}: {str(e)}")
                    raise
                
                model.name = names[i]

                # Load custom hyperparameters if provided
                if configs[i] != "default":
                    hyperparameters_path = base_config_path + str.replace(model_type, "Model", "") + "/" + configs[i]
                    logger.info(f"Loading hyperparameters from: {hyperparameters_path}")
                    try:
                        with open(hyperparameters_path, 'r') as f:
                            hyperparameters = yaml.safe_load(f)
                        model.set_params(hyperparameters)
                        logger.info(f"Hyperparameters loaded successfully. Parameters: {list(hyperparameters.keys())}")
                    except FileNotFoundError:
                        logger.error(f"Hyperparameter file not found: {hyperparameters_path}")
                        raise
                else:
                    logger.info(f"Using default hyperparameters for {names[i]}")

                # Set storage location for model persistence
                if self.save_models:
                    model.storage_location = f"{self.outdir + model.algorithm}_{model.name}"
                    logger.debug(f"Model storage location: {model.storage_location}")
                else:
                    model.storage_location = "Model is not saved"
                    logger.debug("Model will not be persisted (save_models=False)")

                # Add model to dictionary organized by type
                if model_type in models:
                    models[model_type].append(model)
                else:
                    models[model_type] = [model]
                
                logger.info(f"[{current_model_count}/{total_models}] Successfully created model '{names[i]}'")
        
        logger.info("=" * 80)
        logger.info(f"Model creation completed. Total models created: {total_models}")
        for model_type, model_list in models.items():
            logger.info(f"  {model_type}: {len(model_list)} models - {[m.name for m in model_list]}")
        logger.info("=" * 80)
        
        return models

    def load_models(self, needed_models: dict, available_models_dict: dict, model_base_paths: dict) -> dict:
        """
        Loads pre-trained models from disk if they are not already in memory.
        Checks if models are available in the available_models_dict, and loads from disk if needed.
        
        Args:
            needed_models: Dictionary specifying which models are needed:
                {
                    'ModelType': {
                        'Names': ['model_name1', 'model_name2', ...]
                    }
                }
            available_models_dict: Dictionary containing already-loaded models
            model_base_paths: Dictionary mapping model types to their base storage paths
            
        Returns:
            Updated available_models_dict with all needed models loaded
        """
        logger.info("=" * 80)
        logger.info("Loading Models from Configuration")
        logger.info("=" * 80)
        
        total_models_needed = sum(len(needed_models[model_type]["Names"]) for model_type in needed_models)
        models_loaded = 0
        models_already_available = 0
        
        for model_type, model_names in needed_models.items():
            logger.info(f"Processing {len(model_names['Names'])} models of type: {model_type}")
            available_models = available_models_dict[model_type]
            
            # Determine base path for loading models
            base_path = model_base_paths[model_type] if model_base_paths[model_type] != "default" else self.outdir
            logger.debug(f"Base path for {model_type}: {base_path}")
            
            for model_name in model_names["Names"]:
                found = False
                
                # Check if model is already loaded in memory
                for model in available_models:
                    if model_name == model.name:
                        found = True
                        models_already_available += 1
                        logger.info(f"Model '{model_name}' ({model_type}) already available in memory")
                        break
                
                # If not in memory, load from disk
                if not found:
                    logger.info(f"Model '{model_name}' ({model_type}) not in memory. Loading from disk...")
                    try:
                        model = eval(model_type + "Model()")
                        model.name = model_name
                        model.algorithm = model_type
                        logger.debug(f"Loading model from base path: {base_path}")
                        model.load(base_path)
                        available_models_dict[model_type].append(model)
                        models_loaded += 1
                        logger.info(f"Successfully loaded model '{model_name}' ({model_type}) from disk")
                    except Exception as e:
                        logger.error(f"Failed to load model '{model_name}' ({model_type}): {str(e)}")
                        raise
        
        logger.info("=" * 80)
        logger.info("Model Loading Completed")
        logger.info(f"Total models needed: {total_models_needed}")
        logger.info(f"Models loaded from disk: {models_loaded}")
        logger.info(f"Models already in memory: {models_already_available}")
        logger.info("=" * 80)
        
        return available_models_dict