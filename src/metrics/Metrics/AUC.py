from metrics.models.Generic_Models import *
from metrics.Metrics.IMetricSpec import IFloatMetricSpec
from sklearn.metrics import roc_curve, auc


class AUC(IFloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> GenericMetric:
        true_label = metric_parameters['true_labels']
        predicition_probs = metric_parameters['prediction_probs']
        fpr, tpr, _ = roc_curve(true_label, predicition_probs)
        auc_score = auc(fpr, tpr)
        return GenericMetric(metric_name="AUC", metric_value=FloatValue(metric_value=auc_score), metric_spec=AUC())

    def needs_probabilities(self) -> bool:
        return True
