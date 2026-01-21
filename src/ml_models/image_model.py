import torch
import numpy as np
import tensorflow as tf
import gc
import os
import time
import logging
from abc import abstractmethod
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchmetrics import Accuracy, Precision, Recall, Specificity, AUROC, F1Score
from ml_models.model_interface import Model
import csv
import sys

logger = logging.getLogger(__name__)

class ImageModel(Model):
    """Base class for all image classification models with template method pattern"""

    def __init__(self, image_model_parameters, model_name):
        # for explanation of the parameters, see the manual for the config file (doc/Anleitung Config-Datei.md)
        self.num_epochs_pneumonia = image_model_parameters["num_epochs_pneumonia"]
        self.num_epochs_ards = image_model_parameters["num_epochs_ards"]
        self.batch_size_pneumonia = image_model_parameters["batch_size_pneumonia"]
        self.batch_size_ards = image_model_parameters["batch_size_ards"]
        self.SEED_pneumonia = image_model_parameters["SEED_pneumonia"]
        self.SEED_ards = image_model_parameters["SEED_ards"]
        self.learning_rate = image_model_parameters["learning_rate"]
        self.k_folds = image_model_parameters["k_folds"]
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

        self.model = None
        self._storage_location = None

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
        
        self.model = model

    def test_image_model(self, ards_test, info_list):
        print("###############################")
        print("Testing best models")

        # structure of info_list: [DATASET_PNEUMONIA, DATASET_ARDS, MODEL_NAME_CNN, cnn_method, mode n]
        dataset_pneumonia = info_list[0]
        dataset_ards = info_list[1]
        model_name = info_list[2]
        method = info_list[3]
        mode = info_list[4]
        device = self.get_device()
        dataset_name = dataset_pneumonia+'_'+dataset_ards
        
        # find testing models
        test_model_list = [name for name in os.listdir(self.path_results_ards) if name == '{name}_{dataset}_{method}.pt'.format(name=model_name, dataset=dataset_name, method=method)]
        test_dataloader = DataLoader(ards_test, batch_size=self.batch_size_ards, shuffle=True)

        lr = 1e-2
        batch_size = 64

        loss_fn, kfold, optimizer, scheduler = self.get_helpers(self.model)

        for test_model in test_model_list:
            
            # load model
            self.model.load_state_dict(torch.load(os.path.join(self.path_results_ards, test_model), weights_only=False))
            self.model.to(device)
            
            # Testing
            test_results = self.perform_testing(device, test_dataloader, loss_fn, test_model)

            # save results/metrics for testing
            path_test_res = 'test_metrics_{name}_{dataset}_{method}_{mode}.pt'.format(name=model_name, dataset=dataset_name, method=method, mode=mode)
            with open(os.path.join(self.path_results_ards, path_test_res), 'w') as f:
                    w = csv.DictWriter(f, test_results.keys())
                    w.writeheader()
                    w.writerow(test_results)

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
    
    @property
    def storage_location(self):
        return self._storage_location

    @storage_location.setter
    def storage_location(self, location):
        self._storage_location = location