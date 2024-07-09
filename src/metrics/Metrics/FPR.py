from metrics.models.Generic_Models import *
from metrics.Metrics.IMetricSpec import IListMetricSpec
from sklearn.metrics import roc_curve


class FPR(IListMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> GenericMetric:
        prediction_probs = metric_parameters['prediction_probs']
        true_labels = metric_parameters['true_labels']
        fpr, _, _ = roc_curve(true_labels, prediction_probs)
        return GenericMetric(metric_name="FPR", metric_value=ListValue(metric_value=fpr), metric_spec=FPR())


    def needs_probabilities(self) -> bool:
        return True