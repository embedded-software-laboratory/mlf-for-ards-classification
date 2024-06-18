from metrics.Generic_Models import *
from metrics.Metrics.IMetricSpec import IListMetricSpec
from sklearn.metrics import roc_curve


class TPR(IListMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> ListValue:
        prediction_probs = metric_parameters['prediction_probs']
        true_labels = metric_parameters['true_labels']
        _, tpr, _ = roc_curve(true_labels, prediction_probs)
        return ListValue(metric_value=tpr)
