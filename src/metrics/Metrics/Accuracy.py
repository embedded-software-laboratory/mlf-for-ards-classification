from metrics.models.Generic_Models import *

from sklearn.metrics import accuracy_score


class Accuracy(IFloatMetricSpec):

    def calculate_metric(self, metric_params: dict) -> GenericMetric:
        predicted_class = metric_params['predicted_class']
        true_class = metric_params['true_labels']
        return GenericMetric(metric_name="Accuracy",
                             metric_value=FloatValue(metric_value=accuracy_score(true_class, predicted_class)),
                             metric_spec=Accuracy())

    def needs_probabilities(self) -> bool:
        return False
