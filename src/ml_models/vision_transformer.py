from ml_models.image_model import ImageModel
import torch
from processing.datasets import ImageDataset
from torch import nn
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ExponentialLR
import time
import os
import csv
import re
import timm
from torch.cuda.amp import autocast, GradScaler

class VisionTransformer(ImageModel):

    def __init__(self, image_model_parameters, model_name):
        super().__init__(image_model_parameters, model_name)
        self.use_amp = torch.cuda.is_available()  # Enable automatic mixed precision on CUDA
        self.scaler = GradScaler() if self.use_amp else None

    def build_model(self, model_name):
        """
        Function for building a ViT model.
        
        :param model_name: (str) The model name for the ViT model, for ViT typically small/base/large/huge model size and 8/16/32 patch size, 
                                but for this thesis only small model size and patch size 16. 
                                Format: "ViT-small-16" or just "ViT" (defaults to small-16)
        :return: model contains the ViT model which we are using for the classification
                model_name (str) contains the model name which we are using
        """
        name_split = re.split('-', model_name)
        
        # Handle different model name formats with defaults
        if len(name_split) >= 3:
            model_size = name_split[1]
            patch_size = name_split[2]
        elif len(name_split) == 2:
            model_size = name_split[1]
            patch_size = "16"  # default patch size
        else:
            # Just "ViT" - use defaults
            model_size = "small"
            patch_size = "16"
        
        print(f"Building ViT model: size={model_size}, patch_size={patch_size}", flush=True)

        # generate the model name and save existing models
        model_name_og = str("vit_" + model_size + "_patch" + str(patch_size) + "_224.augreg_in21k") # Changed Original: from "_224_in21k"
        timm_models_list = timm.list_models('vit*', pretrained=True)

        # generate model if it exists
        if model_name_og in timm_models_list: 
            model = timm.create_model(model_name_og, pretrained=True, img_size=ImageDataset.get_image_size(None, "VIT")[1], in_chans=1)
        else:
            raise Exception("Pretrained ViT model not supported")

        # reset classifier to number of classes
        model.reset_classifier(2)
        
        # Count trainable parameters for debugging
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"ViT: {trainable:,} trainable / {total:,} total parameters")
            
        print("Selected model: {}".format(model_name))
            
        return model
    
    def get_helpers(self, model):
        loss_fn = nn.CrossEntropyLoss()
        # Handle k_folds=1 as simple train/validation split (no cross-validation)
        if self.k_folds == 1:
            kfold = None  # Signal to use simple split instead of k-fold
        else:
            kfold = KFold(n_splits=self.k_folds, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        scheduler = ExponentialLR(optimizer=optimizer, gamma=0.96)

        return loss_fn, kfold, optimizer, scheduler
    
    def perform_training(self, device, train_dataloader, model, valid_dataloader, loss_fn, optimizer, scheduler, epoch, history, model_name, dataset_name, method, PATH_RESULT_MODEL, PATH_RESULTS, best_acc, best_auroc, mode):
        # Show learning rate and device info
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}, Device: {device}, AMP: {self.use_amp}", flush=True)
        
        if 'cuda' in str(device) and torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}, Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB", flush=True)
        
        # Training
        train_start = time.perf_counter()
        train_loss, correct = self.train_loop(device, train_dataloader, model, loss_fn, optimizer) 
        train_end = time.perf_counter()
        train_accuracy = self.accuracy.compute()
        train_precision = self.precision.compute()
        train_recall = self.recall.compute()
        train_specificity = self.specificity.compute()
        self.accuracy.reset(), self.precision.reset(), self.recall.reset(), self.specificity.reset()

        # Validation
        valid_loss, correct = self.test_loop(device, valid_dataloader, model, loss_fn) 
        valid_accuracy = self.accuracy.compute()
        valid_precision = self.precision.compute()
        valid_recall = self.recall.compute()
        valid_specificity = self.specificity.compute()
        valid_auroc = self.auroc.compute()
        valid_f1 = self.f1.compute()
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
            path = '{name}_{dataset}_{unfreeze_setting}_{pMode}.pt'.format(name=model_name, dataset=dataset_name, unfreeze_setting=method, pMode=mode)
            torch.save(model.state_dict(), os.path.join(PATH_RESULT_MODEL, path))
        if best_auroc < valid_auroc:
            best_auroc = valid_auroc
            path = '{name}_{dataset}_{unfreeze_setting}_{pMode}.pt'.format(name=model_name, dataset=dataset_name, unfreeze_setting=method, pMode=mode)
            torch.save(model.state_dict(), os.path.join(PATH_RESULT_MODEL, path))

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
        path_res = 'metrics_{name}_{dataset}_{unfreeze_setting}.pt'.format(name=model_name, dataset=dataset_name, unfreeze_setting=method)
        with open(os.path.join(PATH_RESULTS, path_res), 'w') as f:
            w = csv.DictWriter(f, history.keys())
            w.writeheader()
            w.writerow(history)

    def perform_testing(self, device, test_dataloader, loss_fn, test_model):
        # Testing
        test_loss, correct = self.test_loop(device, test_dataloader, self.model, loss_fn)
        test_accuracy = self.accuracy.compute()
        test_precision = self.precision.compute()
        test_recall = self.recall.compute()
        test_specificity = self.specificity.compute()
        test_auroc = self.auroc.compute()
        test_f1 = self.f1.compute()
        self.accuracy.reset(), self.precision.reset(), self.recall.reset(), self.specificity.reset(), self.auroc.reset(), self.f1.reset()

        # save metrics in list
        test_results = {'test_loss': [], 'test_acc':[], 'test_prec':[], 'test_recall':[], 'test_specificity':[], 'test_auroc':[], 'test_f1':[]}
        test_results['test_loss'].append(test_loss)
        test_results['test_acc'].append(test_accuracy)
        test_results['test_prec'].append(test_precision)
        test_results['test_recall'].append(test_recall)
        test_results['test_specificity'].append(test_specificity)
        test_results['test_auroc'].append(test_auroc)
        test_results['test_f1'].append(test_f1)

        # show testing metrics
        print("Model:{} \nTest Loss:{:.3f} AVG Test Acc:{:.2f} Test Precision:{:.3f}  Test Recall:{:.3f} Test Specificity:{:.3f} Test AUROC:{:.3f} Test f1-score: {:.3f}\n".format(
                                                                                                                test_model,
                                                                                                                test_loss,
                                                                                                                test_accuracy,
                                                                                                                test_precision,
                                                                                                                test_recall,
                                                                                                                test_specificity,
                                                                                                                test_auroc,
                                                                                                                test_f1))
        
        return test_results

    def train_loop(self, device, dataloader, model, loss_fn, optimizer):
        """
        Generates the training process.
        
        :param device: (str) The device which is being used, wither cuda or cpu
        :param dataloader: (Dataloader) Object for image and label of specific idx which were generated before
        :param model: (timm/pytorch) Model used, which should be vit_small_path16
        :param loss_fn: (CrossEntropyLoss) Loss function used
        :param optimizer: (AdamW) Optimizer used
        :return: train_loss (float) contains the training loss value
                train_correct (int) contains the number of correctly labeled images
        """
        train_loss, train_correct = 0.0, 0
        
        # Train mode on
        model.train()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        batch_start_time = time.perf_counter()
        
        print(f"  Training: Processing {num_batches} batches...", flush=True)
        
        for batch, (X, y) in enumerate(dataloader):    
            
            # load image and label into device with non_blocking for async transfer
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Use automatic mixed precision for faster training on GPUs
            if self.use_amp:
                with autocast():
                    pred = model(X)
                    pred = pred.to(torch.float32)
                    loss = loss_fn(pred, y)
                
                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Compute prediction and loss
                pred = model(X)
                pred = pred.to(torch.float32)
                
                # Compute loss and optimize
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update values
            train_loss += loss.item()
            train_correct += (torch.argmax(pred, axis=1) == y.long().squeeze()).type(torch.float32).sum().item()
            pred = torch.argmax(pred, axis=1).to(torch.float32)
            
            # Update scores for result
            self.accuracy.update(pred.to('cpu'),y.to('cpu'))
            self.precision.update(pred.to('cpu'),y.to('cpu'))
            self.recall.update(pred.to('cpu'),y.to('cpu'))
            self.specificity.update(pred.to('cpu'),y.to('cpu'))
            
            # Debug output every 10 batches or last batch
            if (batch + 1) % 10 == 0 or (batch + 1) == num_batches:
                batch_time = time.perf_counter() - batch_start_time
                avg_batch_time = batch_time / (batch + 1)
                current_loss = train_loss / (batch + 1)
                current_acc = train_correct / ((batch + 1) * X.size(0))
                
                debug_msg = f"    Batch {batch + 1}/{num_batches} - Loss: {current_loss:.4f}, Acc: {current_acc:.3f}, Time: {avg_batch_time:.3f}s/batch"
                
                # Add GPU memory info if using CUDA
                if 'cuda' in str(device) and torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1024**3
                    mem_reserved = torch.cuda.memory_reserved() / 1024**3
                    debug_msg += f", GPU Mem: {mem_allocated:.2f}GB/{mem_reserved:.2f}GB"
                
                print(debug_msg, flush=True)

        train_loss /= num_batches
        train_correct /= size
        print(f"  Training completed: {num_batches} batches in {time.perf_counter() - batch_start_time:.2f}s", flush=True)

        return train_loss, train_correct
    
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
        test_loss, test_correct = 0, 0

        # Evalution/Testing mode on
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        val_start_time = time.perf_counter()
        
        print(f"  Validation: Processing {num_batches} batches...", flush=True)

        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                
                # load image and label into device with non_blocking for async transfer
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
                # Compute prediction and loss
                pred = model(X)
                pred = pred.to(torch.float32)
                test_loss += loss_fn(pred, y).item()
                test_correct += (torch.argmax(pred, axis=1) == y.long().squeeze()).type(torch.float32).sum().item()
                pred = torch.argmax(pred, axis=1).to(torch.float32)
            
                # Update scores for result
                self.accuracy.update(pred.to('cpu'),y.to('cpu'))
                self.precision.update(pred.to('cpu'),y.to('cpu'))
                self.recall.update(pred.to('cpu'),y.to('cpu'))
                self.specificity.update(pred.to('cpu'),y.to('cpu'))
                self.auroc.update(pred.to('cpu'),y.to('cpu'))
                self.f1.update(pred.to('cpu'),y.to('cpu'))
                
                # Debug output every 10 batches or last batch
                if (batch + 1) % 10 == 0 or (batch + 1) == num_batches:
                    current_loss = test_loss / (batch + 1)
                    current_acc = test_correct / ((batch + 1) * X.size(0))
                    print(f"    Batch {batch + 1}/{num_batches} - Loss: {current_loss:.4f}, Acc: {current_acc:.3f}", flush=True)

        test_loss /= num_batches
        test_correct /= size
        print(f"  Validation completed: {num_batches} batches in {time.perf_counter() - val_start_time:.2f}s", flush=True)

        return test_loss, test_correct
        
    def create_model(self, model_name, dataset_name, method, dataset_train, info_list, device, PATH_RESULT_MODEL, mode):
        model = self.build_model(model_name)
        model = self.unfreeze_model(model, method)            
        model.to(device)  
        return model
    
    def unfreeze_model(self, model, unfreeze_setting):
        """
        This function is for Transfer-Learning of the ViT model, it configures the parameters of the layers according 
        to which unfreeze setting we want to use by freezing those we do not need.
        
        :param model: (timm/pytorch) ViT model which was generated before
        :param unfreeze_setting: (str) The layers name which we want to train it on 
        :return: model (timm/pytorch) contains the model with the freezes parameters according to the setting we wanted
        """

        # set all parameters to false or freeze all parameters 
        for param in model.blocks.parameters():
            param.requires_grad = False
            
        # for model unfreeze all parameters or activate all parameters
        if unfreeze_setting == 'model': 
            for param in model.blocks.parameters():
                param.requires_grad = True
            print("Entire model unfreezed")
            
        # freeze all parameters but the last block
        elif unfreeze_setting == 'last_block':
            for param in model.blocks[len(model.blocks)-1].parameters():
                param.requires_grad = True
            print("Last block unfreezed")
            
        # freeze all parameters but the classifier layer
        elif unfreeze_setting == 'classifier': 
            print("Classifier unfreezed")
            raise Exception("It is not yet supported unfreeze the whole classifier.")
        else:
            print(unfreeze_setting)
            raise Exception("It is not yet supported to train this part of the model.")
        
        # Count trainable parameters after unfreezing
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"After unfreezing: {trainable:,} trainable / {total:,} total parameters")

        return model
    
    def get_created_model(self, device, model_name, dataset_pneumonia, method, mode):
        model = self.get_model(device, model_name, dataset_pneumonia, method, mode)   
        model = self.unfreeze_model(model, method)    
        model.to(device)
        return model