from datetime import datetime

from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, matthews_corrcoef, roc_curve, make_scorer
from sklearn.model_selection import StratifiedKFold
from metrics.models.Result import *

import os


class Evaluation:

    def __init__(self, config, model, dataset_training, dataset_test):
        self.config = config

        self.eval_info = EvaluationInformation(config, model)

    def evaluate(self, model, test_data) -> Result:
        feature_data = test_data.loc[:, test_data.columns != 'ards']
        if model.has_predict_proba():
            self.eval_info.predicted_probas = model.predict_proba(feature_data)[:, 1]

        else:
            self.eval_info.predicted_labels = model.predict(feature_data)

        self.eval_info.true_labels = test_data['ards']
        if self.config["process"]["perform_threshold_optimization"]:
            threshold_optimizers = self.config['evaluation']['threshold_optimization_algorithms']
        else:
            threshold_optimizers = ["Standard"]
        optimizer_list = []
        for optimizer in threshold_optimizers:
            eval_result = SplitFactory.factory_method(self.eval_info, "Evaluation", optimizer)
            optimizer_result = OptimizerFactory.factory_method([eval_result], optimizer)
            optimizer_list.append(optimizer_result)

        result = ResultFactory.factory_method(self.eval_info, optimizer_list)
        return result

    def cross_validate(self, model, data) -> Result:
        labels = data["ards"]
        predictors = data.loc[:, data.columns != 'ards']

        # Create Splits for Crossvalidation
        cross_validation = StratifiedKFold(n_splits=self.eval_info.n_splits,
                                           shuffle=self.eval_info.shuffle,
                                           random_state=self.eval_info.random_state)

        if self.config["process"]["perform_threshold_optimization"]:
            threshold_optimizers = self.config['evaluation']['threshold_optimization_algorithms']
        else:
            threshold_optimizers = ["Standard"]

        optimizer_eval_dict = {}
        for optimizer in threshold_optimizers:
            optimizer_eval_dict[optimizer] = []

        for (train_set, test_set), i in zip(cross_validation.split(predictors, labels), range(self.eval_info.n_splits)):
            predictors_train = predictors.iloc[train_set]
            labels_train = labels.iloc[train_set]

            predictors_test = predictors.iloc[test_set]
            labels_test = labels.iloc[test_set]

            # Learn model for the splut
            model.train_model(predictors_train.assign(ards=labels_train))
            if self.config["process"]["save_models"]:
                save_path = self.config["storage_path"] if self.config["storage_path"] else "./Save/" + str(
                    datetime.now().strftime("%m-%d-%Y_%H-%M-%S")) + "/" + model.name + "_split_" + str(i)

                model.save(save_path)

            if model.has_predict_proba():
                self.eval_info.predicted_probas = model.predict_proba(predictors_test.assign(ards=labels_test))[:,1]

            else:
                self.eval_info.prediction_labels = model.predict(predictors_test.assign(ards=labels_test))

            self.eval_info.true_labels = labels_test

            for optimizer in threshold_optimizers:
                eval_result = SplitFactory.factory_method(self.eval_info, f"Evaluation split {i}", optimizer)
                optimizer_eval_dict[optimizer].append(eval_result)
        optimizer_list = []
        for optimizer in threshold_optimizers:
            mean_split = MeanSplitFactory.factory_method(optimizer_eval_dict[optimizer])
            optimizer_eval_dict[optimizer].append(mean_split)
            complete_eval_list = optimizer_eval_dict[optimizer]
            optimizer_result = OptimizerFactory.factory_method(complete_eval_list, optimizer)
            optimizer_list.append(optimizer_result)
        result = ResultFactory.factory_method(self.eval_info, optimizer_list)
        return result


class Evaluation_Old:
    # TODO replace with new approach
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
            'tprs': tpr.tolist(),
            'fprs': fpr.tolist(),
            'auc_scores': auc_score,
            'acc': acc,
            'sens': sensitivity,
            'spec': specificity,
            'f1': f1,
            'mcc': mcc,
            'jaccard': 1,  # TODO: Actual Jaccard
        }

        return metric_dict

    def _compute_roc_auc(self, model, predictors, labels):
        """Function that calculates AUROC, fpr and tpr for a given model on a dataset"""
        #prediction_probs = model.predict(predictors)
        prediction_probs = model.predict_proba(predictors)[:, 1]
        fpr, tpr, thresholds = roc_curve(labels, prediction_probs)
        auc_score = auc(fpr, tpr)
        return fpr, tpr, auc_score

    def _compute_metrics(self, model, predictors, labels):
        """Function that calculates F1, Accuarcy, MCC, Sensitivity and Specificity of a given model on a dataset"""
        predictions = model.predict(predictors)
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        f1 = f1_score(labels, predictions)
        acc = accuracy_score(labels, predictions)
        mcc = matthews_corrcoef(labels, predictions)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
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

    def cross_validate_model(self, model, test_data, outdir):
        """Function that perfroms crossvalidation"""

        labels = test_data["ards"]
        predictors = test_data.loc[:, test_data.columns != 'ards']

        # Create Splits for Crossvalidation
        cross_validation = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=None)

        # Variables for perfromance metrics
        fprs, tprs, scores, sensitivities, specificities, accuracies, mccs, f1s = [], [], [], [], [], [], [], []

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
        mean_acc = sum(accuracies) / len(accuracies)
        mean_sens = sum(sensitivities) / len(sensitivities)
        mean_spec = sum(specificities) / len(specificities)
        mean_f1 = sum(f1s) / len(f1s)
        mean_mcc = sum(mccs) / len(mccs)
        metric_dict = {
            'tprs': tprs,
            'fprs': fprs,
            'auc_scores': scores,
            'acc': mean_acc,
            'sens': mean_sens,
            'spec': mean_spec,
            'f1': mean_f1,
            'mcc': mean_mcc
        }
        return metric_dict

    def _compute_roc_auc_cv(self, index, model, predictors, labels):
        """Function that calculates AUROC, fpr and tpr for a given model on a dataset in crossvalidation"""

        prediction_probs = model.predict_proba(predictors.iloc[index])[:, 1]
        fpr, tpr, thresholds = roc_curve(labels.iloc[index], prediction_probs)
        auc_score = auc(fpr, tpr)
        return fpr, tpr, auc_score

    def _compute_metrics_cv(self, index, model, predictors, labels):
        """Function that calculates F1, Accuarcy, MCC, Sensitivity and Specificity of a given model on a dataset during Crossvalidation"""
        predictions = model.predict(predictors.iloc[index])
        tn, fp, fn, tp = confusion_matrix(labels.iloc[index], predictions).ravel()
        f1 = f1_score(labels.iloc[index], predictions)
        acc = accuracy_score(labels.iloc[index], predictions)
        mcc = matthews_corrcoef(labels.iloc[index], predictions)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return f1, acc, mcc, sensitivity, specificity
