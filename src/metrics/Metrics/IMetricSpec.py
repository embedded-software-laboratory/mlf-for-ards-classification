from metrics.models.Generic_Models import *


class IMetricSpec:

    def calculate_metric(self, metric_parameters: dict) -> GenericValue:
        raise NotImplementedError

    def calculate_metric_mean(self, metric_list: list[GenericMetric]) -> GenericMetric:
        raise NotImplementedError

    def needs_probabilities(self) -> bool:
        raise NotImplementedError


class IFloatMetricSpec(IMetricSpec):

    def __init__(self):
        super().__init__()
        self.metric_spec = IFloatMetricSpec
        self.metric_type = GenericMetric
    def calculate_metric(self, metric_parameters: dict) -> GenericMetric:
        raise NotImplementedError

    def calculate_metric_mean(self, metric_list: list[GenericMetric]) -> GenericMetric:
        metric_value_sum = 0.0
        for metric in metric_list:
            metric_value_sum += metric.metric_value
        return GenericMetric(metric_name=metric_list[0].metric_name, metric_value=metric_value_sum,
                             metric_spec=metric_list[0].metric_spec)


class IIntMetricSpec(IMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> GenericMetric:
        raise NotImplementedError

    def calculate_metric_mean(self, metric_list: list[GenericMetric]) -> GenericMetric:
        metric_value_sum = 0.0
        for metric in metric_list:
            metric_value_sum += metric.metric_value
        average_value = FloatValue(metric_value=metric_value_sum)
        return GenericMetric(metric_name=metric_list[0].metric_name, metric_value=average_value,
                             metric_spec=metric_list[0].metric_spec)


class IListMetricSpec(IMetricSpec):
    def calculate_metric(self, metric_parameters: dict) -> GenericMetric:
        raise NotImplementedError

    def calculate_metric_mean(self, metric_list: list[GenericMetric]) -> GenericMetric:
        average_value = ListValue(metric_value=["Mean calculation makes no sense here"])
        return GenericMetric(metric_name=metric_list[0].metric_name, metric_value=average_value,
                             metric_spec=metric_list[0].metric_spec)

