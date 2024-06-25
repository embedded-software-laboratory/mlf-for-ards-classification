import numpy as np

from IThresholdOptimizer import IThresholdOptimizer


class MaxTPRMinFPR(IThresholdOptimizer):
    def calculate_optimal_threshold(self, tpr: np.ndarray, fpr: np.ndarray, threshold: np.ndarray) -> float:
        optimal_threshold = threshold[np.argmax(tpr - fpr)]
        return optimal_threshold
