import torch
import numpy as np
import tensorflow as tf
import gc
import os
import time
import logging
import json
from abc import abstractmethod
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchmetrics import Accuracy, Precision, Recall, Specificity, AUROC, F1Score, MatthewsCorrCoef
from ml_models.model_interface import Model
import csv
import sys

logger = logging.getLogger(__name__)

class ImageModel(Model):
    """Base class for all image classification models with template method pattern"""

    def __init__(self, image_model_parameters, model_name):
        super().__init__()  # Initialize parent Model class
        
        # Set model metadata
        self.name = model_name
        self.type = 'ImageModel'
        self.algorithm = model_name  # For image models, algorithm is the same as name (ResNet, DenseNet, ViT)
        
        # Determine if this is a ViT or CNN model
        is_vit = 'vit' in model_name.lower()
        is_cnn = ('resnet' in model_name.lower() or 'densenet' in model_name.lower())
        
        # Model-specific parameters
        if is_vit:
            # ViT-specific parameters
            self.learning_rate = image_model_parameters.get("learning_rate_vit", 0.0005)
            self.batch_size = image_model_parameters.get("batch_size_vit", 32)
            self.k_folds = image_model_parameters.get("k_folds_vit", 5)
            # ViT uses same epoch parameters as CNN models
            self.num_epochs_pneumonia = image_model_parameters.get("num_epochs_pneumonia", 20)
            self.num_epochs_ards = image_model_parameters.get("num_epochs_ards", 10)
            self.batch_size_pneumonia = self.batch_size
            self.batch_size_ards = self.batch_size
            print(f"Using ViT-specific parameters: LR={self.learning_rate}, batch_size={self.batch_size}, "
                  f"epochs_pneumonia={self.num_epochs_pneumonia}, epochs_ards={self.num_epochs_ards}")
        elif is_cnn:
            # CNN-specific parameters
            self.learning_rate = image_model_parameters.get("learning_rate_cnn", 0.05)
            self.learning_rate_pre = image_model_parameters.get("learning_rate_pre_cnn", 0.001)
            self.weight_decay = image_model_parameters.get("weight_decay_cnn", 1e-5)
            self.weight_decay_pre = image_model_parameters.get("weight_decay_pre_cnn", 1e-5)
            self.epoch_decay = image_model_parameters.get("epoch_decay_cnn", 0.002)
            self.margin = image_model_parameters.get("margin_cnn", 1.0)
            # Stage-specific parameters
            self.num_epochs_pneumonia = image_model_parameters.get("num_epochs_pneumonia", 20)
            self.num_epochs_ards = image_model_parameters.get("num_epochs_ards", 10)
            self.batch_size_pneumonia = image_model_parameters.get("batch_size_pneumonia", 128)
            self.batch_size_ards = image_model_parameters.get("batch_size_ards", 8)
            self.batch_size_pre = image_model_parameters.get("batch_size_pre_cnn", 64)
            self.k_folds = image_model_parameters.get("k_folds", 5)
            print(f"Using CNN-specific parameters: LR={self.learning_rate}, LR_pre={self.learning_rate_pre}, "
                  f"weight_decay={self.weight_decay}, margin={self.margin}")
        else:
            # Fallback to old behavior
            self.learning_rate = image_model_parameters.get("learning_rate", 0.001)
            self.num_epochs_pneumonia = image_model_parameters["num_epochs_pneumonia"]
            self.num_epochs_ards = image_model_parameters["num_epochs_ards"]
            self.batch_size_pneumonia = image_model_parameters["batch_size_pneumonia"]
            self.batch_size_ards = image_model_parameters["batch_size_ards"]
            self.k_folds = image_model_parameters.get("k_folds", 1)
            print(f"Using generic parameters: LR={self.learning_rate}")
        
        # Common parameters for all models
        self.SEED_pneumonia = image_model_parameters.get("SEED_pneumonia", 123)
        self.SEED_ards = image_model_parameters.get("SEED_ards", 105)
        self.path = image_model_parameters["path"]
        self.model_name = model_name
        self.path_models_pneumonia =  os.path.join(self.path,'models/'+ model_name+'/pneumonia/main')
        self.path_models_ards = os.path.join(self.path,'models/' +  model_name + '/ards/main')
        self.path_results_pneumonia = os.path.join(self.path,'results/' + model_name + '/pneumonia/main')
        self.path_results_ards = os.path.join(self.path,'results/' + model_name + '/ards/main')

        self.accuracy = Accuracy(task="binary", average='macro')
        self.precision = Precision(task="binary", average='macro')
        self.recall = Recall(task="binary", average='macro')
        self.specificity = Specificity(task="binary", average='macro')
        self.f1 = F1Score(task="binary")
        self.auroc = AUROC(task="binary")
        self.mcc = MatthewsCorrCoef(task="binary")

        self.model = None
        self._storage_location = None
        self.training_history = {}  # Store training histories for both stages

    # ==================== PUBLIC API ====================

    def train_image_model(self, pneumonia_train, ards_train, info_list):
        """
        Main training entry point - orchestrates two-stage training.
        Stage 1: Pneumonia training
        Stage 2: ARDS training with transfer learning
        
        Args:
            pneumonia_train: Pneumonia dataset for initial training
            ards_train: ARDS dataset for transfer learning
            info_list: [DATASET_PNEUMONIA, DATASET_ARDS, MODEL_NAME, method, mode]
        """
        dataset_pneumonia = info_list[0]
        dataset_ards = info_list[1]
        model_name = info_list[2]
        method = info_list[3]
        mode = info_list[4]

        # run twice: one time for pneumonia and one time for ards with transfer learning
        for disease in ['PNEUMONIA', 'ARDS']:
            # Handle method parameter (can be single value or list)
            if isinstance(method, list):
                current_method = method[0] if disease == 'PNEUMONIA' else method[1]
            else:
                current_method = method
                
            self._train_single_stage(
                disease=disease,
                pneumonia_train=pneumonia_train,
                ards_train=ards_train,
                dataset_pneumonia=dataset_pneumonia,
                dataset_ards=dataset_ards,
                model_name=model_name,
                method=current_method,
                mode=mode
            )

    # ==================== TEMPLATE METHODS ====================

    def _train_single_stage(self, disease, pneumonia_train, ards_train, 
                           dataset_pneumonia, dataset_ards, model_name, method, mode):
        """Template method for single training stage"""
        
        # Setup parameters based on disease
        dataset_train = pneumonia_train if disease == 'PNEUMONIA' else ards_train
        dataset_name = dataset_pneumonia if disease == 'PNEUMONIA' else f"{dataset_pneumonia}_{dataset_ards}"
        num_epochs = self.num_epochs_pneumonia if disease == 'PNEUMONIA' else self.num_epochs_ards
        batch_size = self.batch_size_pneumonia if disease == 'PNEUMONIA' else self.batch_size_ards
        SEED = self.SEED_pneumonia if disease == 'PNEUMONIA' else self.SEED_ards
        PATH_MODEL = self.path_models_pneumonia if disease == 'PNEUMONIA' else self.path_models_ards
        PATH_RESULTS = self.path_results_pneumonia if disease == 'PNEUMONIA' else self.path_results_ards
        
        # Handle augmentation naming for ARDS
        dataset_name_og = dataset_name
        if disease == 'ARDS' and mode in ['mode3', 'mode4']:
            dataset_name = f"{dataset_name}_aug"
        
        # Create directories
        os.makedirs(PATH_MODEL, exist_ok=True)
        os.makedirs(PATH_RESULTS, exist_ok=True)
        
        # Setup environment
        self.run_gpu()
        device = self.get_device()
        self.set_all_seeds(SEED)
        
        print("SETUP:", flush=True)
        print(f"model: {model_name}, dataset: {dataset_name_og}, LR: {self.learning_rate}, "
              f"batch_size: {batch_size}, num_epochs: {num_epochs}, k_folds: {self.k_folds}", flush=True)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Create/load model (subclass-specific)
        info_list_full = [dataset_pneumonia, dataset_ards, model_name, method, mode]
        if disease == 'PNEUMONIA':
            model = self.create_model(model_name, dataset_name, method, 
                                     dataset_train, info_list_full, 
                                     device, PATH_MODEL, mode)
        else:
            model = self.get_created_model(device, model_name, dataset_pneumonia, method, mode)
        
        # Get training helpers (subclass-specific: loss, optimizer, etc.)
        loss_fn, kfold, optimizer, scheduler = self.get_helpers(model)
        
        # Check if already trained
        model_path = f'{model_name}_{dataset_name}_{method}.pt'
        if os.path.isfile(os.path.join(PATH_MODEL, model_path)):
            print(f"Training already succeeded for {disease}.")
            self.model = model
            return
        
        print("")
        print("###############################", flush=True)
        print(f"Starting Training for {disease}", flush=True)
        
        history = {
            'epoch': [], 'train_loss': [], 'valid_loss': [],
            'train_acc': [], 'valid_acc': [], 'train_prec': [], 'valid_prec': [],
            'train_recall': [], 'valid_recall': [], 'train_specificity': [], 'valid_specificity': [],
            'valid_auroc': [], 'valid_f1': [], 'train_time': []
        }
        
        best_acc, best_auroc = 0.0, 0.0
        
        # Optimize DataLoader settings for faster training
        num_workers = min(8, os.cpu_count() or 4)  # Use up to 8 workers or available CPUs
        pin_memory = 'cuda' in str(device)  # Only use pin_memory for CUDA devices
        
        # Handle k_folds=1 as simple train/validation split (80/20)
        if kfold is None:
            print("Using simple 80/20 train/validation split (k_folds=1)", flush=True)
            print("###############################", flush=True)
            
            # Simple 80/20 split
            dataset_size = len(dataset_train)
            train_size = int(0.8 * dataset_size)
            val_size = dataset_size - train_size
            
            # Create indices for train and validation
            indices = list(range(dataset_size))
            np.random.seed(SEED)
            np.random.shuffle(indices)
            train_idx = indices[:train_size]
            val_idx = indices[train_size:]
            
            train_loader = DataLoader(dataset_train, batch_size=batch_size, 
                                     sampler=SubsetRandomSampler(train_idx),
                                     num_workers=num_workers, pin_memory=pin_memory,
                                     persistent_workers=(num_workers > 0))
            valid_loader = DataLoader(dataset_train, batch_size=batch_size, 
                                     sampler=SubsetRandomSampler(val_idx),
                                     num_workers=num_workers, pin_memory=pin_memory,
                                     persistent_workers=(num_workers > 0))
            
            model.to(device)
            
            for epoch in range(num_epochs):
                print(f"Epoch {epoch+1}\n-------------------------------", flush=True)
                self.perform_training(device, train_loader, model, valid_loader, loss_fn, 
                                    optimizer, scheduler, epoch, history, model_name, 
                                    dataset_name, method, PATH_MODEL, PATH_RESULTS, 
                                    best_acc, best_auroc, mode)
            
            # Show final validation metrics
            avg_acc = np.mean(history['valid_acc'])
            avg_loss = np.mean(history['valid_loss'])
            print(f"Final validation scores\n-------------------------------", flush=True)
            print(f'> Accuracy: {avg_acc}')
            print(f'> Loss: {avg_loss}\n-------------------------------', flush=True)
            
            # Save training history to JSON
            self._save_training_history(disease, history, model_name, dataset_name, method, mode, PATH_RESULTS)
        else:
            # K-fold cross validation
            for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset_train)):
                print(f"FOLD {fold+1}", flush=True)
                print("###############################", flush=True)
                
                train_loader = DataLoader(dataset_train, batch_size=batch_size, 
                                         sampler=SubsetRandomSampler(train_idx),
                                         num_workers=num_workers, pin_memory=pin_memory,
                                         persistent_workers=(num_workers > 0))
                valid_loader = DataLoader(dataset_train, batch_size=batch_size, 
                                         sampler=SubsetRandomSampler(val_idx),
                                         num_workers=num_workers, pin_memory=pin_memory,
                                         persistent_workers=(num_workers > 0))
                
                model.to(device)
                
                for epoch in range(num_epochs):
                    print(f"Epoch {epoch+1}\n-------------------------------", flush=True)
                    self.perform_training(device, train_loader, model, valid_loader, loss_fn, 
                                        optimizer, scheduler, epoch, history, model_name, 
                                        dataset_name, method, PATH_MODEL, PATH_RESULTS, 
                                        best_acc, best_auroc, mode)
                
                # Show average of validation
                avg_acc = np.mean(history['valid_acc'])
                avg_loss = np.mean(history['valid_loss'])
                print(f"Average scores\n-------------------------------", flush=True)
                print(f'> Accuracy: {avg_acc}')
                print(f'> Loss: {avg_loss}\n-------------------------------', flush=True)
            
            # Save training history to JSON
            self._save_training_history(disease, history, model_name, dataset_name, method, mode, PATH_RESULTS)
        
        self.model = model

    def _save_training_history(self, disease, history, model_name, dataset_name, method, mode, PATH_RESULTS):
        """Save training history to JSON file and store in model's training_history dict"""
        # Convert numpy/torch values to native Python types for JSON serialization
        history_json = {}
        for key, values in history.items():
            if isinstance(values, list):
                history_json[key] = [float(v) if hasattr(v, 'item') else v for v in values]
            else:
                history_json[key] = values
        
        # Save to individual JSON file
        history_filename = f'training_history_{model_name}_{dataset_name}_{method}_{mode}_{disease}.json'
        history_path = os.path.join(PATH_RESULTS, history_filename)
        
        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=4)
        
        logger.info(f"Training history saved to: {history_path}")
        
        # Store in model's training_history dict for later aggregation
        stage_key = f"{disease}_{dataset_name}_{method}_{mode}"
        self.training_history[stage_key] = history_json

    def test_image_model(self, ards_test, info_list):
        logger.info("=" * 70)
        logger.info("TESTING IMAGE MODEL")
        logger.info("=" * 70)

        # structure of info_list: [DATASET_PNEUMONIA, DATASET_ARDS, MODEL_NAME_CNN, cnn_method, mode n]
        dataset_pneumonia = info_list[0]
        dataset_ards = info_list[1]
        model_name = info_list[2]
        method = info_list[3]
        mode = info_list[4]
        device = self.get_device()
        dataset_name = dataset_pneumonia+'_'+dataset_ards
        
        logger.info(f"Model: {model_name}")
        logger.info(f"Pneumonia dataset: {dataset_pneumonia}")
        logger.info(f"ARDS dataset: {dataset_ards}")
        logger.info(f"Method: {method}")
        logger.info(f"Mode: {mode}")
        logger.info(f"Test dataset size: {len(ards_test)} samples")
        logger.info(f"Batch size: {self.batch_size_ards}")
        
        # find testing models
        test_model_pattern = '{name}_{dataset}_{method}_{mode}.pt'.format(name=model_name, dataset=dataset_name, method=method, mode=mode)
        test_model_list = [name for name in os.listdir(self.path_models_ards) if name == test_model_pattern]
        
        if not test_model_list:
            logger.warning(f"No trained model found matching pattern: {test_model_pattern}")
            logger.warning(f"Searched in: {self.path_models_ards}")
            return
        
        logger.info(f"Found {len(test_model_list)} trained model(s) to test")
        
        test_dataloader = DataLoader(ards_test, batch_size=self.batch_size_ards, shuffle=True)
        logger.info(f"Test dataloader created with {len(test_dataloader)} batches")

        lr = 1e-2
        batch_size = 64

        loss_fn, kfold, optimizer, scheduler = self.get_helpers(self.model)

        for idx, test_model in enumerate(test_model_list, 1):
            logger.info("-" * 70)
            logger.info(f"Testing model {idx}/{len(test_model_list)}: {test_model}")
            
            # load model
            model_path = os.path.join(self.path_models_ards, test_model)
            logger.info(f"Loading model from: {model_path}")
            self.model.load_state_dict(torch.load(model_path, weights_only=False))
            self.model.to(device)
            logger.info(f"Model loaded and moved to device: {device}")
            
            # Testing
            logger.info("Starting model testing on ARDS test data...")
            test_results = self.perform_testing(device, test_dataloader, loss_fn, test_model)
            
            # Log the computed metrics
            logger.info("=" * 70)
            logger.info("TEST RESULTS:")
            logger.info("=" * 70)
            if isinstance(test_results['test_loss'], list):
                logger.info(f"  Loss:        {test_results['test_loss'][0]:.4f}")
                logger.info(f"  Accuracy:    {test_results['test_acc'][0]:.4f}")
                logger.info(f"  Precision:   {test_results['test_prec'][0]:.4f}")
                logger.info(f"  Recall:      {test_results['test_recall'][0]:.4f}")
                logger.info(f"  Specificity: {test_results['test_specificity'][0]:.4f}")
                logger.info(f"  AUROC:       {test_results['test_auroc'][0]:.4f}")
                logger.info(f"  F1-Score:    {test_results['test_f1'][0]:.4f}")
                logger.info(f"  MCC:         {test_results['test_mcc'][0]:.4f}")
            else:
                logger.info(f"  Loss:        {test_results['test_loss']:.4f}")
                logger.info(f"  Accuracy:    {test_results['test_acc']:.4f}")
                logger.info(f"  Precision:   {test_results['test_prec']:.4f}")
                logger.info(f"  Recall:      {test_results['test_recall']:.4f}")
                logger.info(f"  Specificity: {test_results['test_specificity']:.4f}")
                logger.info(f"  AUROC:       {test_results['test_auroc']:.4f}")
                logger.info(f"  F1-Score:    {test_results['test_f1']:.4f}")
                logger.info(f"  MCC:         {test_results['test_mcc']:.4f}")
            logger.info("=" * 70)

            # save results/metrics for testing
            path_test_res = 'test_metrics_{name}_{dataset}_{method}_{mode}.pt'.format(name=model_name, dataset=dataset_name, method=method, mode=mode)
            output_path = os.path.join(self.path_results_ards, path_test_res)
            with open(output_path, 'w') as f:
                    w = csv.DictWriter(f, test_results.keys())
                    w.writeheader()
                    w.writerow(test_results)
            logger.info(f"Test metrics saved to: {output_path}")
        
        logger.info("=" * 70)
        logger.info("TESTING COMPLETED")
        logger.info("=" * 70)

    def run_gpu(self):
        """Function to make sure GPU is running"""
        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=10240)]
                )
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs", flush=True)
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e, flush=True)

    def get_device(self):
        """Function to get training device"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using {device} device", flush=True)
        
        return device
    
    def set_all_seeds(self, SEED):
        """Function for reproducibility"""
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    
    def get_model(self, device, model_name, dataset_name, method, mode):
        """
        Function for getting the pretrained model for transfer learning with pneumonia
        
        :param device: (str) The device which is being used, either cuda or cpu
        :param model_name: (str) The models name used to generate model
        :return: model is the model generated
        """
        
        # generate model
        model = self.build_model(model_name)
        model.to(device)

        # load model weights saved in state dict
        print(dataset_name)
        path = '{name}_{dataset}_{method}_{mode}.pt'.format(name=model_name, dataset=dataset_name, method=method, mode=mode)
        state_dict = torch.load(os.path.join(self.path_models_pneumonia, path), map_location=device, weights_only=False)
        model.load_state_dict(state_dict)

        return model
    
    # ==================== ABSTRACT METHODS (Subclass must implement) ====================
    
    @abstractmethod
    def build_model(self, model_name):
        """Build the neural network architecture"""
        raise NotImplementedError
    
    @abstractmethod
    def get_helpers(self, model):
        """Return (loss_fn, kfold, optimizer, scheduler)"""
        raise NotImplementedError
    
    @abstractmethod
    def perform_training(self, device, train_dataloader, model, valid_dataloader, loss_fn, optimizer, 
                        scheduler, epoch, history, model_name, dataset_name, method, PATH_RESULT_MODEL, 
                        PATH_RESULTS, best_acc, best_auroc, mode):
        """Perform one epoch of training and validation"""
        raise NotImplementedError
    
    @abstractmethod
    def perform_testing(self, device, test_dataloader, loss_fn, test_model):
        """Perform testing on test dataset"""
        raise NotImplementedError
    
    @abstractmethod
    def create_model(self, model_name, dataset_name, method, dataset_train, info_list, device, PATH_RESULT_MODEL, mode):
        """Create model for stage 1 (can include pretraining)"""
        raise NotImplementedError
    
    @abstractmethod
    def get_created_model(self, device, model_name, dataset_pneumonia, method, mode):
        """Load model for stage 2 (transfer learning)"""
        raise NotImplementedError
    
    # ==================== UTILITY METHODS ====================
    
    def has_predict_proba(self):
        """
        Image models output probabilities (sigmoid/softmax), so they support predict_proba.
        Returns True for all image models.
        """
        return True
    
    @property
    def storage_location(self):
        return self._storage_location

    @storage_location.setter
    def storage_location(self, location):
        self._storage_location = location