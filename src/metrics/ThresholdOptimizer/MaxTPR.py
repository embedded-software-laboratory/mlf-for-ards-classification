import numpy as np

from IThresholdOptimizer import IThresholdOptimizer


class MaxTPR(IThresholdOptimizer):
    def calculate_optimal_threshold(self, tpr: list[float], fpr: list[float], threshold) -> float:

        return threshold[np.argmax(tpr)]
