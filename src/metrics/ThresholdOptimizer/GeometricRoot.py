import numpy as np

from IThresholdOptimizer import IThresholdOptimizer


class GeometricRoot(IThresholdOptimizer):
    def calculate_optimal_threshold(self, tpr: np.ndarray, fpr: np.ndarray, threshold: np.ndarray) -> float:

        return threshold[np.argmax(tpr*(1-fpr))]
