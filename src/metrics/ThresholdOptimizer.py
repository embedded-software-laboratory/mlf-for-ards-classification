from numpy import ndarray
import numpy as np


class IThresholdOptimizer:

    def calculate_optimal_threshold(self, tpr: ndarray, fpr: ndarray, threshold: ndarray) -> float:
        raise NotImplementedError


class Standard(IThresholdOptimizer):
    def calculate_optimal_threshold(self, tpr: ndarray, fpr: ndarray, threshold: ndarray) -> float:
        return 0.5


class MaxTPRMinFPR(IThresholdOptimizer):
    def calculate_optimal_threshold(self, tpr: ndarray, fpr: ndarray, threshold: ndarray) -> float:
        print("TPR")
        print(tpr)
        print("FPR")
        print(fpr)
        print("Difference:")
        print(tpr-fpr)
        print(np.argmax(tpr-fpr))
        print("Threshold")
        print(threshold)
        optimal_threshold = threshold[np.argmax(tpr - fpr)]
        print(optimal_threshold)
        return optimal_threshold


class MaxTPR(IThresholdOptimizer):
    def calculate_optimal_threshold(self, tpr: ndarray, fpr: ndarray, threshold: ndarray) -> float:
        return threshold[np.argmax(tpr)]


class GeometricRoot(IThresholdOptimizer):
    def calculate_optimal_threshold(self, tpr: ndarray, fpr: ndarray, threshold: ndarray) -> float:
        return threshold[np.argmax(tpr * (1 - fpr))]
