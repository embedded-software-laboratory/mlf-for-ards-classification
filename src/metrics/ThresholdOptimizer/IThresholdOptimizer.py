

class IThresholdOptimizer:

    def calculate_optimal_threshold(self, tpr: list[float], fpr: list[float], threshold: list[float]) -> float:
        raise NotImplementedError