from metrics.models.Generic_Models import *
from metrics.Metrics.IMetricSpec import IFloatMetricSpec
from sklearn.metrics import f1_score


class F1Score(IFloatMetricSpec):

    def calculate_metric(self, metric_params: dict) -> GenericMetric:
        predicted_class = metric_params['predicted_label']
        true_class = metric_params['true_labels']
        return GenericMetric(metric_name="F1-Score",
                             metric_value=FloatValue(metric_value=f1_score(predicted_class, true_class)),
                             metric_spec=F1Score())

    def needs_probabilities(self) -> bool:
        return False
