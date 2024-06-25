from numpy import ndarray


class IThresholdOptimizer:

    def calculate_optimal_threshold(self, tpr: ndarray, fpr: ndarray, threshold: ndarray) -> float:
        raise NotImplementedError
