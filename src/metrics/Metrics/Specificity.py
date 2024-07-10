from metrics.models.Generic_Models import *
from metrics.Metrics.IMetricSpec import IFloatMetricSpec
from sklearn.metrics import confusion_matrix


class Specificity(IFloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> GenericMetric:
        prediction_labels = metric_parameters['predicted_label']
        true_labels = metric_parameters['true_labels']
        tn, fp, fn, tp = confusion_matrix(true_labels, prediction_labels).ravel()

        return GenericMetric(metric_name="Specificity",
                             metric_value=FloatValue(metric_value=(tn / (tn + fp))),
                             metric_spec=Specificity())

    def needs_probabilities(self) -> bool:
        return False
