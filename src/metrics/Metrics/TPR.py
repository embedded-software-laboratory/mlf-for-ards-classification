from metrics.models.Generic_Models import *
from metrics.Metrics.IMetricSpec import IListMetricSpec
from sklearn.metrics import roc_curve


class TPR(IListMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> GenericMetric:
        prediction_probs = metric_parameters['prediction_probs']
        true_labels = metric_parameters['true_labels']
        _, tpr, _ = roc_curve(true_labels, prediction_probs)
        return GenericMetric(metric_name="TPR", metric_value=ListValue(metric_value=tpr), metric_spec=TPR())

    def needs_probabilities(self) -> bool:
        return True
