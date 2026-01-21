import torch
import numpy as np
import tensorflow as tf
import gc
import os
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchmetrics import Accuracy, Precision, Recall, Specificity, AUROC, F1Score
from ml_models.model_interface import Model
import csv
import sys

class ImageModel(Model):

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


    def train_image_model(self, pneumonia_train, ards_train, info_list):
        # structure of info_list: [DATASET_PNEUMONIA, DATASET_ARDS, MODEL_NAME_CNN, cnn_method, mode n]
        dataset_pneumonia = info_list[0]
        dataset_ards = info_list[1]
        model_name = info_list[2]
        method = info_list[3]
        mode = info_list[4]

        # run twice: one time for pneumonia and one time for ards with transfer learning
        for disease in ['PNEUMONIA', 'ARDS']:
            method = info_list[3][0] if disease == 'PNEUMONIA' else info_list[3][1]
            dataset_train = pneumonia_train if disease == 'PNEUMONIA' else ards_train
            dataset_name = dataset_pneumonia if disease == 'PNEUMONIA' else dataset_pneumonia+'_'+dataset_ards
            num_epochs = self.num_epochs_pneumonia if disease == 'PNEUMONIA' else self.num_epochs_ards 
            batch_size = self.batch_size_pneumonia if disease == 'PNEUMONIA' else self.batch_size_ards
            SEED = self.SEED_pneumonia if disease == 'PNEUMONIA' else self.SEED_ards
            PATH_RESULT_MODEL = self.path_models_pneumonia if disease == 'PNEUMONIA' else self.path_models_ards
            PATH_RESULTS = self.path_results_pneumonia if disease == 'PNEUMONIA' else self.path_results_ards

            if not os.path.isdir(PATH_RESULT_MODEL):
                os.makedirs(PATH_RESULT_MODEL)
            if not os.path.isdir(PATH_RESULTS):
                os.makedirs(PATH_RESULTS)

            # save dataset_name without augmentation label in next line to use the name for test
            dataset_name_og = dataset_name
            
            # add augmentation to the dataset_name if training data is augmented for ARDS dataset
            if disease == 'ARDS':
                if (mode == 'mode3') or (mode == 'mode4'):
                    dataset_name = dataset_name+'_aug'
                else:
                    dataset_name = dataset_name
        
            # start operations by setting cpu, gpu and seeds
            self.run_gpu()
            device = self.get_device()
            self.set_all_seeds(SEED)

            print("SETUP:", flush=True)
            print("model: {}, dataset: {}, LR: {}, batch_size: {}, num_epochs: {}, k_folds: {}".format(model_name, dataset_name_og, self.learning_rate, batch_size, num_epochs, self.k_folds), flush=True)

            torch.cuda.empty_cache()
            gc.collect() 

            if disease == 'PNEUMONIA':
                model = self.create_model(model_name, dataset_name, method, dataset_train, info_list, device, PATH_RESULT_MODEL, mode)
            else:
                model = self.get_created_model(device, model_name, dataset_pneumonia, method, mode)

            loss_fn, kfold, optimizer, scheduler = self.get_helpers(model)

            # only train, if the specific training was not already done before and saved in the folder
            path = '{name}_{dataset}_{method}.pt'.format(name=model_name, dataset=dataset_name, method=method)
            if not os.path.isfile(os.path.join(PATH_RESULT_MODEL, path)):
                print("")
                print("###############################", flush=True)
                print("Starting Training for "+ disease, flush=True)

                history = {'epoch':[],'train_loss': [], 'valid_loss': [],'train_acc':[],'valid_acc':[],'train_prec':[], 'valid_prec':[],'train_recall':[], 'valid_recall':[], 'train_specificity':[], 'valid_specificity':[], 'valid_auroc':[], 'valid_f1':[], 'train_time': []}
                
                best_acc, best_auroc = 0., 0.
                # run training and validation for n folds
                for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset_train)):
                    print(f"FOLD {fold+1}", flush=True)
                    print("###############################", flush=True)

                    # split training data into training and validation
                    train_subsampler = SubsetRandomSampler(train_idx)
                    valid_subsampler = SubsetRandomSampler(val_idx)
                    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, sampler=train_subsampler)
                    valid_dataloader = DataLoader(dataset_train, batch_size=batch_size, sampler=valid_subsampler)

                    model.to(device)

                    for epoch in range(num_epochs):
                        print(f"Epoch {epoch+1}\n-------------------------------", flush=True)
                        self.perform_training(device, train_dataloader, model, valid_dataloader, loss_fn, optimizer, scheduler, epoch, history, model_name, dataset_name, method, PATH_RESULT_MODEL, PATH_RESULTS, best_acc, best_auroc, mode)

                    # show average of validation
                    avg_acc = np.mean(history['valid_acc'])
                    avg_loss = np.mean(history['valid_loss'])
                    print(f"Average scores \n-------------------------------", flush=True)
                    print(f'> Accuracy: {avg_acc}')
                    print(f'> Loss: {avg_loss} \n-------------------------------', flush=True)
            else:
                print("Training alrady succeded.")

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
            self.model.load_state_dict(torch.load(os.path.join(self.path_results_ards, test_model)))
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
        Function for getting the pretrained ViT model for transfer learning with pneumonia
        
        :param device: (str) The device which is being used, wither cuda or cpu
        :param model_name: (str) The models name used to generate model
        :return: model is the model generated
        """
        
        # generate model
        model = self.build_model(model_name)
        model.to(device)

        # load model weights saved in state dict
        print(dataset_name)
        path = '{name}_{dataset}_{method}_{mode}.pt'.format(name=model_name, dataset=dataset_name, method=method, mode=mode)
        state_dict = torch.load(os.path.join(self.path_models_pneumonia, path), map_location=device)
        model.load_state_dict(state_dict)

        return model
    
    def build_model(self, model_name):
        raise NotImplementedError
    
    def get_helpers(self, model):
        raise NotImplementedError
    
    def perform_training(self, device, train_dataloader, model, valid_dataloader, loss_fn, optimizer, scheduler, epoch, history, model_name, dataset_name, method, PATH_RESULT_MODEL, PATH_RESULTS, best_acc, best_auroc, mode):
        raise NotImplementedError
    
    def perform_testing(self, device, test_dataloader, loss_fn, test_model):
        raise NotImplementedError
    
    def create_model(self, model_name, dataset_name, method, dataset_train, info_list, device, PATH_RESULT_MODEL, mode):
        raise NotImplementedError
    
    def get_created_model(self, device, model_name, dataset_pneumonia, method, mode):
        raise NotImplementedError
    
    @property
    def storage_location(self):
        return self._storage_location

    @storage_location.setter
    def storage_location(self, location):
        self._storage_location = location