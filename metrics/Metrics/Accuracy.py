from metrics.Generic_Models import *
class Accuracy(GenericMetric):
    metric_name = 'Accuracy'

    def calculate_metric(self, predictions, ground_truth):