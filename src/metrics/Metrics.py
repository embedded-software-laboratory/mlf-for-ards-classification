import metrics
from metrics.Models import GenericValue, GenericMetric, ListValue, FloatValue, StringValue, IntValue
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, roc_curve, f1_score, auc

from metrics.Models import *





class Accuracy(FloatMetricSpec):

    def calculate_metric(self, metric_params: dict) -> metrics.GenericMetric:
        predicted_class = metric_params['predicted_label']
        true_class = metric_params['true_labels']
        return metrics.GenericMetric(metric_name="Accuracy",
                             metric_value=FloatValue(metric_value=accuracy_score(true_class, predicted_class)),
                             metric_spec=Accuracy())

    def needs_probabilities(self) -> bool:
        return False


class AUC(FloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> GenericMetric:
        true_label = metric_parameters['true_labels']
        predicition_probs = metric_parameters['prediction_probs']
        fpr, tpr, _ = roc_curve(true_label, predicition_probs)
        auc_score = auc(fpr, tpr)
        return GenericMetric(metric_name="AUC", metric_value=FloatValue(metric_value=auc_score), metric_spec=AUC())

    def needs_probabilities(self) -> bool:
        return True


class F1Score(FloatMetricSpec):

    def calculate_metric(self, metric_params: dict) -> GenericMetric:
        predicted_class = metric_params['predicted_label']
        true_class = metric_params['true_labels']
        return GenericMetric(metric_name="F1-Score",
                             metric_value=FloatValue(metric_value=f1_score(predicted_class, true_class)),
                             metric_spec=F1Score())

    def needs_probabilities(self) -> bool:
        return False


class FPR(ListMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> GenericMetric:
        prediction_probs = metric_parameters['prediction_probs']
        true_labels = metric_parameters['true_labels']
        fpr, _, _ = roc_curve(true_labels, prediction_probs)
        return GenericMetric(metric_name="FPR", metric_value=ListValue(metric_value=fpr), metric_spec=FPR())

    def needs_probabilities(self) -> bool:
        return True


class MCC(FloatMetricSpec):

    def calculate_metric(self, metric_params: dict) -> GenericMetric:
        predicted_class = metric_params['predicted_label']
        true_class = metric_params['true_labels']
        return GenericMetric(metric_name="MCC",
                             metric_value=FloatValue(metric_value=matthews_corrcoef(true_class, predicted_class)),
                             metric_spec=MCC())

    def needs_probabilities(self) -> bool:
        return False


class OptimalProbability(FloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> GenericMetric:
        fpr = metric_parameters['fpr']
        tpr = metric_parameters['tpr']
        threshold = metric_parameters['thresholds']
        calc_func = metric_parameters['calc_func']
        optimal_prob = calc_func(fpr, tpr, threshold)
        return GenericMetric(metric_name="Optimal Decision Threshold",
                             metric_value=FloatValue(metric_value=optimal_prob),
                             metric_spec=OptimalProbability())

    def needs_probabilities(self) -> bool:
        return True


class Sensitivity(FloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> GenericMetric:
        prediction_labels = metric_parameters['predicted_label']
        true_labels = metric_parameters['true_labels']
        tn, fp, fn, tp = confusion_matrix(true_labels, prediction_labels).ravel()

        return GenericMetric(metric_name="Sensitivity",
                             metric_value=FloatValue(metric_value=(tp / (tp + fn))),
                             metric_spec=Sensitivity())

    def needs_probabilities(self) -> bool:
        return False


class Specificity(FloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> GenericMetric:
        prediction_labels = metric_parameters['predicted_label']
        true_labels = metric_parameters['true_labels']
        tn, fp, fn, tp = confusion_matrix(true_labels, prediction_labels).ravel()

        return GenericMetric(metric_name="Specificity",
                             metric_value=FloatValue(metric_value=(tn / (tn + fp))),
                             metric_spec=Specificity())

    def needs_probabilities(self) -> bool:
        return False


class TPR(ListMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> GenericMetric:
        prediction_probs = metric_parameters['prediction_probs']
        true_labels = metric_parameters['true_labels']
        _, tpr, _ = roc_curve(true_labels, prediction_probs)
        return GenericMetric(metric_name="TPR", metric_value=ListValue(metric_value=tpr), metric_spec=TPR())

    def needs_probabilities(self) -> bool:
        return True
