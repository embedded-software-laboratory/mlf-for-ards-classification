from metrics.Generic_Models import *
from metrics.Metrics.IMetricSpec import IFloatMetricSpec
from sklearn.metrics import confusion_matrix


class Sensitivity(IFloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> FloatValue:
        prediction_labels = metric_parameters['prediction_labels']
        true_labels = metric_parameters['true_labels']
        tn, fp, fn, tp = confusion_matrix(true_labels, prediction_labels).ravel()

        return FloatValue(metric_value=(tp / (tp + fn)))
