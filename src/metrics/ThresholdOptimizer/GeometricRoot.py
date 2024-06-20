import numpy as np

from IThresholdOptimizer import IThresholdOptimizer


class GeometricRoot(IThresholdOptimizer):
    def calculate_optimal_threshold(self, tpr: list[float], fpr: list[float], threshold: list[float]) -> float:

        return threshold[optimal_idx]
