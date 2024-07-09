from metrics.models.Generic_Models import *
from metrics.Metrics.IMetricSpec import IFloatMetricSpec


class OptimalProbability(IFloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> GenericMetric:

        fpr = metric_parameters['fpr']
        tpr = metric_parameters['tpr']
        threshold = metric_parameters['thresholds']
        calc_func = metric_parameters['calc_func']
        optimal_prob = calc_func(fpr, tpr, threshold)
        print(isinstance(OptimalProbability, IFloatMetricSpec))
        return GenericMetric(metric_name="Optimal Decision Threshold",
                             metric_value=FloatValue(metric_value=optimal_prob),
                             metric_spec=OptimalProbability)

    def needs_probabilities(self) -> bool:
        return True
