from metrics import *
from metrics.ThresholdOptimizer.IThresholdOptimizer import IThresholdOptimizer


class Standard(IThresholdOptimizer):
    def calculate_optimal_threshold(self, tpr: list[float], fpr: list[float], threshold: list[float]) -> float:
        return 0.5
