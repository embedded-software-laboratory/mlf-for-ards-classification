
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, matthews_corrcoef, roc_curve, make_scorer
from sklearn.model_selection import StratifiedKFold

import os

class Evaluation:

    def __init__(self, timeseries_classes, config):
        #cross validation params
        self.n_splits = config["cross_validation"]["n_splits"]
        self.shuffle = config["cross_validation"]["shuffle"]
        self.timeseries_classes = timeseries_classes

    def evaluate(self, model, test_data):
        """Function used to evalute a model"""

        labels = test_data["ards"]
        predictors = test_data.loc[:, test_data.columns != 'ards']
        
        # Compute evaluation results
        fpr, tpr, auc_score = self._compute_roc_auc(model, predictors, labels) 
        f1, acc, mcc, sensitivity, specificity = self._compute_metrics(model, predictors, labels)

        # Store results in file
        metric_dict = {
            'tprs' : tpr.tolist(),
            'fprs' : fpr.tolist(),
            'auc_scores' : auc_score,
            'acc' : acc, 
            'sens' : sensitivity,
            'spec' : specificity,
            'f1' : f1,
            'mcc' : mcc,
            'jaccard': 1, # TODO: Actual Jaccard
        }

        return metric_dict
    
    def _compute_roc_auc(self,  model, predictors, labels) :
        """Function that calculates AUROC, fpr and tpr for a given model on a dataset"""
        #prediction_probs = model.predict(predictors)
        prediction_probs = model.predict_proba(predictors)[:,1]
        fpr, tpr , thresholds= roc_curve(labels, prediction_probs) 
        auc_score = auc(fpr, tpr)
        return  fpr, tpr ,auc_score
    
    def _compute_metrics(self, model, predictors, labels) :
        """Function that calculates F1, Accuarcy, MCC, Sensitivity and Specificity of a given model on a dataset"""
        predictions = model.predict(predictors)
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        f1 = f1_score(labels, predictions)
        acc = accuracy_score(labels, predictions)
        mcc = matthews_corrcoef(labels, predictions)
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        return f1, acc, mcc, sensitivity, specificity
    
    def perform_cross_validation(self, data, outdir):
        # dictionary instead of array
        results = {}
        for modelClass in self.timeseries_classes:
            model = modelClass()
            result = self.cross_validate_model(model, data, outdir)
            # add data under model name in dict
            results[model.name] = result
        return results
    
    def cross_validate_model(self, model, test_data, outdir) :
        """Function that perfroms crossvalidation"""

        labels = test_data["ards"]
        predictors = test_data.loc[:, test_data.columns != 'ards']

        # Create Splits for Crossvalidation
        cross_validation = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=None)

        # Variables for perfromance metrics
        fprs, tprs, scores, sensitivities, specificities, accuracies, mccs, f1s = [], [], [], [],[], [], [], []
        
        # Perform crossvalidation
        for (train_set, test_set), i in zip(cross_validation.split(predictors, labels), range(self.n_splits)):
            predictors_train = predictors.iloc[train_set]
            labels_train = labels.iloc[train_set]

            # Learn model for the splut
            model.train_model(predictors_train.assign(ards=labels_train))
            target_dir = outdir + "cross_validation/"
            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)
            model.save(target_dir + model.name + "_" + str(i))

            # Evaluation of model
            _, _, auc_score_training = self._compute_roc_auc_cv(train_set, model, predictors, labels) 
            fpr, tpr, auc_score_test = self._compute_roc_auc_cv(test_set, model, predictors, labels) 
            f1, acc, mcc, sensitivity, specificity = self._compute_metrics_cv(test_set, model, predictors, labels)
            scores.append((auc_score_training, auc_score_test))
            fprs.append(fpr.tolist())
            tprs.append(tpr.tolist())
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            accuracies.append(acc)
            
            mccs.append(mcc)
            f1s.append(f1)
        mean_acc = sum(accuracies)/len(accuracies)
        mean_sens = sum(sensitivities)/len(sensitivities)
        mean_spec = sum(specificities) / len(specificities)
        mean_f1 = sum(f1s)/ len(f1s)
        mean_mcc = sum(mccs)/len(mccs)
        metric_dict = {
            'tprs' : tprs,
            'fprs' : fprs,
            'auc_scores' : scores,
            'acc' : mean_acc, 
            'sens' : mean_sens,
            'spec' : mean_spec,
            'f1' : mean_f1,
            'mcc' : mean_mcc
        }
        return metric_dict
    
    def _compute_roc_auc_cv(self, index, model, predictors, labels) :
        """Function that calculates AUROC, fpr and tpr for a given model on a dataset in crossvalidation"""
        
        prediction_probs = model.predict_proba(predictors.iloc[index])[:,1]
        fpr, tpr , thresholds= roc_curve(labels.iloc[index], prediction_probs) 
        auc_score = auc(fpr, tpr)
        return  fpr, tpr ,auc_score
    
    def _compute_metrics_cv(self, index, model, predictors, labels) :
        """Function that calculates F1, Accuarcy, MCC, Sensitivity and Specificity of a given model on a dataset during Crossvalidation"""
        predictions = model.predict(predictors.iloc[index])
        tn, fp, fn, tp = confusion_matrix(labels.iloc[index], predictions).ravel()
        f1 = f1_score(labels.iloc[index], predictions)
        acc = accuracy_score(labels.iloc[index], predictions)
        mcc = matthews_corrcoef(labels.iloc[index], predictions)
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        return f1, acc, mcc, sensitivity, specificity