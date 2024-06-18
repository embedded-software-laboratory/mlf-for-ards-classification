from metrics.GenericMetric import GenericMetric
from metrics.Generic_Models import GenericValue, FloatValue, IntValue, ListValue, StringValue


class IMetricSpec:

    def calculate_metric(self, metric_parameters: dict) -> GenericValue:
        raise NotImplementedError

    def calculate_metric_mean(self, average_parmeters: dict) -> GenericValue:
        raise NotImplementedError


class IFloatMetricSpec(IMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> FloatValue:
        raise NotImplementedError

    def calculate_metric_mean(self, average_parameters: dict) -> FloatValue:
        metric_value_sum = 0.0
        for _, value in average_parameters.items():
            metric_value_sum += value
        return FloatValue(metric_value=metric_value_sum / len(average_parameters))


class IIntMetricSpec(IMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> IntValue:
        raise NotImplementedError

    def calculate_metric_mean(self, average_parameters: dict) -> FloatValue:
        metric_value_sum = 0.0
        for _, value in average_parameters.items():
            metric_value_sum += value
        return FloatValue(metric_value=metric_value_sum / len(average_parameters))


class IListMetricSpec(IMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> ListValue:
        raise NotImplementedError

    def calculate_metric_mean(self, average_parameters: dict) -> StringValue:
        return StringValue(metric_value="Mean calculation makes no sense")
