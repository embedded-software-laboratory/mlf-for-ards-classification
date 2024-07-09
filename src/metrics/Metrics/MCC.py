from metrics.models.Generic_Models import *
from metrics.Metrics.IMetricSpec import IFloatMetricSpec
from sklearn.metrics import matthews_corrcoef


class MCC(IFloatMetricSpec):

    def calculate_metric(self, metric_params: dict) -> GenericMetric:
        predicted_class = metric_params['predicted_class']
        true_class = metric_params['true_class']
        return GenericMetric(metric_name="MCC",
                             metric_value=FloatValue(metric_value=matthews_corrcoef(true_class, predicted_class)),
                             metric_spec=MCC())

    def needs_probabilities(self) -> bool:
        return False
