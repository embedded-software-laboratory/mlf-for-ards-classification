from metrics.Generic_Models import *
from metrics.Metrics.IMetricSpec import IFloatMetricSpec
from sklearn.metrics import accuracy_score


class Accuracy(IFloatMetricSpec):

    def calculate_metric(self, metric_params: dict) -> FloatValue:
        predicted_class = metric_params['predicted_class']
        true_class = metric_params['true_class']
        return FloatValue(metric_value=accuracy_score(predicted_class, true_class))
