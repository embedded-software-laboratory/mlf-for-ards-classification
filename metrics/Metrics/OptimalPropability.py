from metrics.Generic_Models import *
from metrics.Metrics.IMetricSpec import IFloatMetricSpec
from sklearn.metrics import roc_curve


class OptimalProbability(IFloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> FloatValue:
        thresholds = metric_parameters['thresholds']
        true_labels = metric_parameters['true_labels']
        prediction_probs= metric_parameters['prediction_probabilities']
        calc_func = metric_parameters['calc_func']

        fpr, tpr, _ = roc_curve(true_labels, prediction_probs)
        # TODO implement finding of optimal prob
        return FloatValue(metric_value=optimal_prob)
