from metrics.Generic_Models import *
from metrics.Metrics.IMetricSpec import IFloatMetricSpec
from sklearn.metrics import roc_curve, auc

class AUC(IFloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> FloatValue:
        true_label = metric_parameters['true_label']
        predicition_probs = metric_parameters['prediction_probs']
        fpr, tpr, _ = roc_curve(true_label, predicition_probs)
        auc_score = auc(fpr, tpr)
        return FloatValue(metric_value=auc_score)
