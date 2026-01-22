import torch
from ml_models.image_model import ImageModel
from torch import nn
import torchvision
from libauc.losses import AUCMLoss, CrossEntropyLoss
from sklearn.model_selection import KFold
from libauc.optimizers import PESG, Adam
import time
import os
import csv 
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast, GradScaler

class ResNet50(nn.Module):
    """Class for generating the CNN model ResNet50"""
    
    def __init__(self):
        """
        This init function generates the model RestNet50 with its initial variables with torchvision
        """
        
        # generate resenet50 with torchvision
        super(ResNet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        
        # Modify first conv layer to accept 1-channel (grayscale) input instead of 3-channel (RGB)
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # freeze all layer except for fully connected layer
        for param in self.resnet50.parameters():
            param.requires_grad = False
        
        # Unfreeze the modified first conv layer so it can learn from 1-channel input
        for param in self.resnet50.conv1.parameters():
            param.requires_grad = True

        # define average pool layer
        self.resnet50.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # define fully connected layer which is the final layer
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )
        
        # Ensure classifier is trainable
        for param in self.resnet50.fc.parameters():
            param.requires_grad = True
        
        self.name = 'ResNet50'
        
        # Count trainable parameters for debugging
        trainable = sum(p.numel() for p in self.resnet50.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.resnet50.parameters())
        print(f"{self.name}: {trainable:,} trainable / {total:,} total parameters")

    def forward(self, x):
        """
        This forward function returns the model for further use of ResNet50
        """  
        x = self.resnet50(x)        
        return x
    
class DenseNet121(nn.Module):
    """Class for generating the CNN model DenseNet121"""

    def __init__(self):
        """
        This init function generates the model DenseNet121 with its initial variables with torchvision
        """
        
        # generate densenet121 model with torchvision
        super().__init__()
        model_ft = torchvision.models.densenet121(pretrained=True)
        
        # Modify first conv layer to accept 1-channel (grayscale) input instead of 3-channel (RGB)
        model_ft.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # freeze all layer except for fully connected layer
        for param in model_ft.parameters():
            param.requires_grad = False
        
        # Unfreeze the modified first conv layer so it can learn from 1-channel input
        for param in model_ft.features.conv0.parameters():
            param.requires_grad = True

        # define classifier layer BEFORE setting avgpool to avoid issues
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )
        
        # Ensure classifier is trainable
        for param in model_ft.classifier.parameters():
            param.requires_grad = True
        
        # define average pool layer
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.model = model_ft
        self.name = 'DenseNet121'
        
        # Count trainable parameters for debugging
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"{self.name}: {trainable:,} trainable / {total:,} total parameters")
        
    def forward(self, x):
        """
        This forward function returns the model for further use of DenseNet121
        """
        
        x = self.model(x)
        
        return x

class CNN(ImageModel):

    def __init__(self, image_model_parameters, model_name):
        super().__init__(image_model_parameters, model_name)
        # Note: margin, epoch_decay, weight_decay, learning_rate_pre, batch_size_pre, weight_decay_pre
        # are now set in the parent ImageModel.__init__ from config
        # Set defaults only if not already set (for backward compatibility)
        if not hasattr(self, 'margin'):
            self.margin = 1.0
        if not hasattr(self, 'epoch_decay'):
            self.epoch_decay = 2e-3
        if not hasattr(self, 'weight_decay'):
            self.weight_decay = 1e-5
        if not hasattr(self, 'learning_rate_pre'):
            self.learning_rate_pre = 1e-3
        if not hasattr(self, 'batch_size_pre'):
            self.batch_size_pre = 64
        if not hasattr(self, 'weight_decay_pre'):
            self.weight_decay_pre = 1e-5
        self.use_amp = torch.cuda.is_available()  # Enable automatic mixed precision on CUDA
        self.scaler = GradScaler() if self.use_amp else None

    def create_model(self, model_name, dataset_name, method, dataset_train, info_list, device, PATH_RESULT_MODEL, mode):
        """
        Function for building a CNN model
        
        :param model_name: (str) The model name for the CNN model, for CNN ResNet50 or DenseNet121 and 
        :return: model contains the CNN model which we are using for the classification
                model_name (str) contains the model name which we are using
        """
        pretrain_path = '{name}_{dataset}_{method}_{mode}_pretrained.pt'.format(name=model_name, dataset=dataset_name, method=method, mode=mode)
        if not os.path.isfile(os.path.join(PATH_RESULT_MODEL, pretrain_path)):
            self.pretraining(device, dataset_train, info_list)
        
        base_model = self.build_model(model_name)
        base_model = self.unfreeze_model(base_model, model_name, method)
        base_model = base_model.to(device)
        path = '{name}_{dataset}_{method}_{mode}_pretrained.pt'.format(name=model_name, dataset=dataset_name, method=method, mode=mode)
        base_model.load_state_dict(torch.load(os.path.join(PATH_RESULT_MODEL, path), weights_only=False), strict=False)

        return base_model
    
    def build_model(self, model_name):
        """
        Function for building a CNN model
        
        :param model_name: (str) The model name for the CNN model, for CNN ResNet50 or DenseNet121 and 
        :return: model contains the CNN model which we are using for the classification
                model_name (str) contains the model name which we are using
        """
        
        # build model for CNN with resnet50 or densenet121 (case-insensitive)
        model_name_lower = model_name.lower()
        if 'resnet' in model_name_lower:
            base_model = ResNet50()
        elif 'densenet' in model_name_lower:
            base_model = DenseNet121()
        else:
            raise ValueError(f"Unknown CNN model type: {model_name}. Expected 'ResNet' or 'DenseNet'.")
                
        print("Selected model: {}".format(model_name))

        return base_model
    
    def unfreeze_model(self, model, model_name, unfreeze_setting):
        """
        This function configures which layers to train for transfer learning of CNN models.
        
        :param model: (nn.Module) The CNN model (ResNet50 or DenseNet121)
        :param model_name: (str) The model name to determine architecture
        :param unfreeze_setting: (str) The layers to train: 'last_layer', 'last_block', or 'model'
        :return: model with frozen/unfrozen parameters according to the setting
        """
        model_name_lower = model_name.lower()
        
        # First freeze all parameters
        if 'resnet' in model_name_lower:
            for param in model.resnet50.parameters():
                param.requires_grad = False
        elif 'densenet' in model_name_lower:
            for param in model.model.parameters():
                param.requires_grad = False
        
        if unfreeze_setting == 'last_layer':
            # Only train the classifier/fc layer
            if 'resnet' in model_name_lower:
                for param in model.resnet50.fc.parameters():
                    param.requires_grad = True
                print("ResNet: Only final classifier (fc) layer unfrozen")
            elif 'densenet' in model_name_lower:
                for param in model.model.classifier.parameters():
                    param.requires_grad = True
                print("DenseNet: Only final classifier layer unfrozen")
                
        elif unfreeze_setting == 'last_block':
            # Unfreeze the last layer block + classifier
            if 'resnet' in model_name_lower:
                # Unfreeze layer4 (last residual block) + fc
                for param in model.resnet50.layer4.parameters():
                    param.requires_grad = True
                for param in model.resnet50.fc.parameters():
                    param.requires_grad = True
                print("ResNet: Last residual block (layer4) + classifier unfrozen")
            elif 'densenet' in model_name_lower:
                # Unfreeze denseblock4 (last dense block) + classifier
                for param in model.model.features.denseblock4.parameters():
                    param.requires_grad = True
                for param in model.model.classifier.parameters():
                    param.requires_grad = True
                print("DenseNet: Last dense block (denseblock4) + classifier unfrozen")
                
        elif unfreeze_setting == 'model':
            # Unfreeze all parameters
            if 'resnet' in model_name_lower:
                for param in model.resnet50.parameters():
                    param.requires_grad = True
                print("ResNet: All layers unfrozen")
            elif 'densenet' in model_name_lower:
                for param in model.model.parameters():
                    param.requires_grad = True
                print("DenseNet: All layers unfrozen")
        else:
            raise ValueError(f"Unknown unfreeze_setting '{unfreeze_setting}'. Use 'last_layer', 'last_block', or 'model'.")
        
        # Count trainable parameters after unfreezing
        if 'resnet' in model_name_lower:
            trainable = sum(p.numel() for p in model.resnet50.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.resnet50.parameters())
        elif 'densenet' in model_name_lower:
            trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.model.parameters())
        
        print(f"After unfreezing: {trainable:,} trainable / {total:,} total parameters")
        
        return model
    
    def get_helpers(self, model):
        loss_fn = AUCMLoss()
        # Handle k_folds=1 as simple train/validation split (no cross-validation)
        if self.k_folds == 1:
            kfold = None  # Signal to use simple split instead of k-fold
        else:
            kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        optimizer = PESG(model.parameters(), loss_fn=loss_fn, lr=self.learning_rate, margin=self.margin, epoch_decay=self.epoch_decay, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96)

        return loss_fn, kfold, optimizer, scheduler
    
    def perform_training(self, device, train_dataloader, model, valid_dataloader, loss_fn, optimizer, scheduler, epoch, history, model_name, dataset_name, method, PATH_RESULT_MODEL, PATH_RESULTS, best_acc, best_auroc, mode):
        # Show learning rate and device info
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}, Device: {device}, AMP: {self.use_amp}", flush=True)
        
        if 'cuda' in str(device) and torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}, Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB", flush=True)
        
        # Training
        train_start = time.perf_counter()
        train_loss, correct, train_prediction, train_true = self.train_loop(device, train_dataloader, model, loss_fn, optimizer) #training
        train_end = time.perf_counter()
        train_accuracy = self.accuracy(train_true, train_prediction)
        train_precision = self.precision(train_true, train_prediction)
        train_recall = self.recall(train_true, train_prediction)
        train_specificity = self.specificity(train_true, train_prediction)
        self.accuracy.reset(), self.precision.reset(), self.recall.reset(), self.specificity.reset()

        # Validation
        valid_loss, correct, valid_prediction, valid_true = self.test_loop(device, valid_dataloader, model, loss_fn) #validation
        valid_accuracy = self.accuracy(valid_true, valid_prediction)
        valid_precision = self.precision(valid_true, valid_prediction)
        valid_recall = self.recall(valid_true, valid_prediction)
        valid_specificity = self.specificity(valid_true, valid_prediction)
        valid_auroc = self.auroc(valid_true, valid_prediction)
        valid_f1 = self.f1(valid_true, valid_prediction)
        self.accuracy.reset(), self.precision.reset(), self.recall.reset(), self.specificity.reset(), self.auroc.reset(), self.f1.reset()
        scheduler.step()

        # save in a list for metrics
        history['epoch'].append(epoch+1)
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['train_acc'].append(train_accuracy)
        history['valid_acc'].append(valid_accuracy)
        history['train_prec'].append(train_precision)
        history['valid_prec'].append(valid_precision)
        history['train_recall'].append(train_recall)
        history['valid_recall'].append(valid_recall)
        history['train_specificity'].append(train_specificity)
        history['valid_specificity'].append(valid_specificity)
        history['valid_auroc'].append(valid_auroc)
        history['valid_f1'].append(valid_f1)
        history['train_time'].append(train_end-train_start)

        # save and replace for best model
        if best_acc < valid_accuracy:
            best_acc = valid_accuracy
            path = '{name}_{dataset}_{method}_{mode}.pt'.format(name=model_name, dataset=dataset_name, method=method, mode=mode)
            torch.save(model.state_dict(), os.path.join(PATH_RESULT_MODEL, path))
        if best_auroc < valid_auroc:
            best_auroc = valid_auroc
            path = '{name}_{dataset}_{method}_{mode}.pt'.format(name=model_name, dataset=dataset_name, method=method, mode=mode)
            torch.save(model.state_dict(), os.path.join(PATH_RESULT_MODEL, path))

        # show metrics
        print("Training Loss:{:.3f} AVG Training Acc:{:.2f} Valid Loss:{:.3f} AVG Valid Acc:{:.2f} Valid Precision:{:.3f} Valid Recall:{:.3f} Valid Specificity:{:.3f} Valid AUROC:{:.3f} Best ACC:{:.3f}, Train Time:{:0.4f}\n".format(
                                                                                                    train_loss,
                                                                                                    train_accuracy,
                                                                                                    valid_loss,
                                                                                                    valid_accuracy,
                                                                                                    valid_precision,
                                                                                                    valid_recall,
                                                                                                    valid_specificity,
                                                                                                    valid_auroc,
                                                                                                    best_acc,
                                                                                                    train_end-train_start), flush=True)

        # save results/metrics of training
        path_res = 'metrics_{name}_{dataset}_{method}_{mode}.pt'.format(name=model_name, dataset=dataset_name, method=method, mode=mode)
        with open(os.path.join(PATH_RESULTS,path_res), 'w') as f:
            w = csv.DictWriter(f, history.keys())
            w.writeheader()
            w.writerow(history)  

    def perform_testing(self, device, test_dataloader, loss_fn, test_model):
        test_loss, correct, test_prediction, test_true = self.test_loop(device, test_dataloader, self.model, loss_fn) #testing
        test_accuracy = self.accuracy(test_true, test_prediction)
        test_precision = self.precision(test_true, test_prediction)
        test_recall = self.recall(test_true, test_prediction)
        test_specificity = self.specificity(test_true, test_prediction)
        test_auroc = self.auroc(test_true, test_prediction)
        test_f1 = self.f1(test_true, test_prediction)
        test_mcc = self.mcc(test_true, test_prediction)
        self.accuracy.reset(), self.precision.reset(), self.recall.reset(), self.specificity.reset(), self.auroc.reset(), self.f1.reset(), self.mcc.reset()

        # save metrics in list
        test_results = {'test_loss': [], 'test_acc':[], 'test_prec':[], 'test_recall':[], 'test_specificity':[], 'test_auroc':[], 'test_f1':[], 'test_mcc':[]}
        test_results['test_loss'].append(test_loss)
        test_results['test_acc'].append(test_accuracy)
        test_results['test_prec'].append(test_precision)
        test_results['test_recall'].append(test_recall)
        test_results['test_specificity'].append(test_specificity)
        test_results['test_auroc'].append(test_auroc)
        test_results['test_f1'].append(test_f1)
        test_results['test_mcc'].append(test_mcc)

        # show testing metrics
        print("Model:{} \nTest Loss:{:.3f} AVG Test Acc:{:.2f} Test Precision:{:.3f}  Test Recall:{:.3f} Test Specificity:{:.3f} Test AUROC:{:.3f} Test f1-score: {:.3f} Test MCC: {:.3f}\n".format(
                                                                                                                test_model,
                                                                                                                test_loss,
                                                                                                                test_accuracy,
                                                                                                                test_precision,
                                                                                                                test_recall,
                                                                                                                test_specificity,
                                                                                                                test_auroc,
                                                                                                                test_f1,
                                                                                                                test_mcc))
        return test_results


    def train_loop(self, device, dataloader, model, loss_fn, optimizer):
        """
        Generates the training process.
        
        :param device: (str) The device which is being used, wither cuda or cpu
        :param dataloader: (Dataloader) Object for image and label of specific idx which were generated before
        :param model: (Object) Either ResNet50 or DenseNet121 object which is generated in build_model()
        :param loss_fn: (AUCMLoss) Loss Function used
        :param optimizer: (PESG) Optimizer Function used
        :return: train_loss (float) contains the training loss value
                train_correct (int) contains the number of correctly labeled images
                train_prediction (list) contains the list of predicted label for the classification
                train_true (list) contains the list of true labels of the images
        """
        train_true = torch.tensor([])
        train_prediction = torch.tensor([])
        train_loss, train_correct = 0.0, 0
        
        #train mode on
        model.train()
        train_loss = 0
        index = 0
        total_samples = 0  # Track actual number of samples processed
        total_batches = len(dataloader)
        batch_start_time = time.perf_counter()
        
        print(f"  Training: Processing {total_batches} batches...", flush=True)
        
        for images, labels in dataloader: 

            # load images and labels and run prediction of classification
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Use automatic mixed precision for faster training on GPUs
            if self.use_amp:
                with autocast():
                    output = model(images)
                    output = output.to(torch.float32)
                    # Keep labels on GPU - reshape without unnecessary CPU transfer
                    labels_reshaped = labels.view(-1, 1).to(torch.float32)
                    loss = loss_fn(output, labels_reshaped)
                
                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                output = model(images)
                output = output.to(torch.float32)
                # Keep labels on GPU - reshape without unnecessary CPU transfer
                labels_reshaped = labels.view(-1, 1).to(torch.float32)
                loss = loss_fn(output, labels_reshaped)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            
            # compute prediction by threshold 0.5
            predictions = (output.data >= 0.5).to(torch.float32)

            # compute number of correctly labeled images and load predictions and true values
            train_correct += torch.sum(predictions == labels_reshaped).item()
            train_prediction = torch.cat((train_prediction, predictions.cpu()),0)   
            train_true = torch.cat((train_true, labels_reshaped.cpu()),0)
            total_samples += images.size(0)  # Track actual samples in this batch
            index += 1
            
            # Debug output every 10 batches or last batch
            if index % 10 == 0 or index == total_batches:
                batch_time = time.perf_counter() - batch_start_time
                avg_batch_time = batch_time / index
                current_loss = train_loss / index
                current_acc = train_correct / total_samples  # Use actual sample count
                
                debug_msg = f"    Batch {index}/{total_batches} - Loss: {current_loss:.4f}, Acc: {current_acc:.3f}, Time: {avg_batch_time:.3f}s/batch"
                
                # Add GPU memory info if using CUDA
                if 'cuda' in str(device) and torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    debug_msg += f", GPU Mem: {mem_allocated:.2f}GB/{mem_reserved:.2f}GB"
                
                print(debug_msg, flush=True)

        # compoute training loss
        train_loss = train_loss / index
        print(f"  Training completed: {index} batches in {time.perf_counter() - batch_start_time:.2f}s", flush=True)
        
        return train_loss, train_correct, train_prediction.to(torch.int32), train_true.to(torch.int32)
    

    def test_loop(self, device, dataloader, model, loss_fn):
        """
        Generates the validation or testing process.
        
        :param device: (str) The device which is being used, wither cuda or cpu
        :param dataloader: (Dataloader) Object for image and label of specific idx which were generated before
        :param model: (Object) Either ResNet50 or DenseNet121 object which is generated in build_model()
        :param loss_fn: (AUCMLoss) Loss Function used
        :return: test_loss (float) contains the validation/testing loss value
                test_correct (int) contains the number of correctly labeled images
                test_prediction (list) contains the list of predicted label for the classification
                test_true (list) contains the list of true labels of the images
        """
        test_true = torch.tensor([])
        test_prediction = torch.tensor([])
        test_loss, test_correct = 0.0, 0
        
        # evalution mode on
        model.eval()
        index = 0
        total_samples = 0  # Track actual number of samples processed
        total_batches = len(dataloader)
        val_start_time = time.perf_counter()
        
        print(f"  Validation: Processing {total_batches} batches...", flush=True)
        
        with torch.no_grad():  # Disable gradient computation for validation
            for images, labels in dataloader:
                
                # load images and labels and run prediction of classification
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                output = model(images)
                output = output.to(torch.float32)
                # Keep labels on GPU - reshape without unnecessary CPU transfer
                labels_reshaped = labels.view(-1, 1).to(torch.float32)
                loss = loss_fn(output, labels_reshaped)
                test_loss += loss.item()
                
                # compute prediction by threshold 0.5
                predictions = (output.data >= 0.5).to(torch.float32)
                
                # compute number of correctly labeled images and load predictions and true values
                test_prediction = torch.cat((test_prediction, predictions.cpu()),0)    
                test_true = torch.cat((test_true, labels_reshaped.cpu()),0)
                test_correct += torch.sum(predictions == labels_reshaped).item()
                total_samples += images.size(0)  # Track actual samples in this batch
                index += 1
                
                # Debug output every 10 batches or last batch
                if index % 10 == 0 or index == total_batches:
                    current_loss = test_loss / index
                    current_acc = test_correct / total_samples  # Use actual sample count
                    print(f"    Batch {index}/{total_batches} - Loss: {current_loss:.4f}, Acc: {current_acc:.3f}", flush=True)
            
        test_loss = test_loss / index
        print(f"  Validation completed: {index} batches in {time.perf_counter() - val_start_time:.2f}s", flush=True)
        
        return test_loss, test_correct, test_prediction.to(torch.int32), test_true.to(torch.int32)
    

    def pretraining(self, device, dataset, info_list):
        """
        This function is for Pre-Training the models.
        
        :param dataset: (Dataset) Pneumonia dataset for training and validation
        :param info_list: (list) With following structure [DATASET_PNEUMONIA, DATASET_ARDS, MODEL_NAME_CNN, cnn_method, mode n]
        """

        start_time = time.time()
        print('###############################', flush=True)
        print("Starting Pretraining", flush=True)
        
        # structure of info_list: [DATASET_PNEUMONIA, DATASET_ARDS, MODEL_NAME_CNN, cnn_method, mode n]
        dataset_name = info_list[0]
        model_name = info_list[2]
        method = info_list[3]
        mode = info_list[4]

        # build model
        model = self.build_model(model_name)
        model = model.to(device)

        # Define loss & optimizer
        loss_fn = torch.nn.BCELoss()
        optimizer = Adam(model.parameters(), lr=self.learning_rate_pre, weight_decay=self.weight_decay_pre)

        # definition of random dataloader with optimization settings
        size_train = int(len(dataset) * 0.8)
        dataset_test, dataset_train = random_split(dataset, [len(dataset) - size_train , size_train], generator=torch.Generator().manual_seed(42))
        # Optimized DataLoader: more workers, pin_memory for faster GPU transfer, persistent_workers to keep workers alive
        num_workers = min(8, os.cpu_count() or 4)  # Use up to 8 workers or available CPUs
        train_dataloader =  DataLoader(dataset_train, batch_size=32, num_workers=num_workers, 
                                       shuffle=True, pin_memory=True, persistent_workers=True)
        valid_dataloader = DataLoader(dataset_test, batch_size=32, num_workers=num_workers, 
                                     shuffle=True, pin_memory=True, persistent_workers=True)

        best_val_auc = 0 
        # run training and validation for one epoch
        for epoch in range(1):
            for batch_train, (X, y) in enumerate(train_dataloader):
                
                # activate training mode
                model.train()

                # load image and label to gpu device with non_blocking for async transfer
                X, y  = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                
                # compute prediction and keep on device
                pred = model(X)
                pred = pred.to(torch.float32)
                y_reshaped = y.view(-1, 1).to(torch.float32)

                # compute loss and optimize
                loss = loss_fn(pred, y_reshaped)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # activate validation mode
                model.eval()
                with torch.no_grad():    
                    pred_val_list = []
                    y_val_list = [] 
                    for batch_val, (X_val, y_val) in enumerate(valid_dataloader):
                        
                        # load validation image and label into gpu device
                        X_val,  y_val = X_val.to(device), y_val.to(device)

                        # compute validation prediction
                        pred_val = model(X_val)
                        pred_val_list.append(pred_val.to('cpu').detach().numpy())
                        y_val_list.append(y_val.to('cpu').numpy())

                    # concatenate validation prediction into lists
                    y_val_list = np.concatenate(y_val_list)
                    pred_val_list = np.concatenate(pred_val_list)
                    
                    # compute the auroc score for predicted values
                    val_auc_mean =  roc_auc_score(y_val_list, pred_val_list) 

                    # save the model with best auroc value in the batch
                    if best_val_auc < val_auc_mean:
                        best_val_auc = val_auc_mean
                        path = '{name}_{dataset}_{method}_{mode}_pretrained.pt'.format(name=model_name, dataset=dataset_name, method=method, mode=mode)
                        torch.save(model.state_dict(), os.path.join(self.path_models_pneumonia, path))

                    print ('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, batch_train, val_auc_mean, best_val_auc ))

    def get_created_model(self, device, model_name, dataset_pneumonia, method, mode):
        """Load pretrained model for ARDS training (transfer learning)"""
        return self.get_model(device, model_name, dataset_pneumonia, method, mode)
            
