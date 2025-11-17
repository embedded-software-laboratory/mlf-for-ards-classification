from numpy import ndarray
import numpy as np
import logging

logger = logging.getLogger(__name__)


class IThresholdOptimizer:
    """
    Interface for threshold optimizers.
    Implementations should return a single float threshold given arrays of tpr, fpr and thresholds.
    """

    def calculate_optimal_threshold(self, tpr: ndarray, fpr: ndarray, threshold: ndarray) -> float:
        logger.debug("IThresholdOptimizer.calculate_optimal_threshold called on base class")
        raise NotImplementedError


class Standard(IThresholdOptimizer):
    """
    Standard optimizer: returns fixed threshold 0.5
    """

    def calculate_optimal_threshold(self, tpr: ndarray, fpr: ndarray, threshold: ndarray) -> float:
        logger.debug("Standard optimizer selected - returning 0.5")
        return 0.5


class MaxTPRMinFPR(IThresholdOptimizer):
    """
    Chooses threshold that maximizes (TPR - FPR).
    """

    def calculate_optimal_threshold(self, tpr: ndarray, fpr: ndarray, threshold: ndarray) -> float:
        logger.debug("MaxTPRMinFPR.calculate_optimal_threshold started")
        if len(tpr) == 0 or len(fpr) == 0 or len(threshold) == 0:
            logger.warning("Empty arrays provided to MaxTPRMinFPR - returning 0.5")
            return 0.5
        scores = tpr - fpr
        idx = int(np.argmax(scores))
        optimal_threshold = float(threshold[idx])
        logger.debug(f"MaxTPRMinFPR selected index {idx} with score {scores[idx]:.6f}, threshold {optimal_threshold:.6f}")
        return optimal_threshold


class MaxTPR(IThresholdOptimizer):
    """
    Chooses threshold that maximizes TPR.
    """

    def calculate_optimal_threshold(self, tpr: ndarray, fpr: ndarray, threshold: ndarray) -> float:
        logger.debug("MaxTPR.calculate_optimal_threshold started")
        if len(tpr) == 0 or len(threshold) == 0:
            logger.warning("Empty arrays provided to MaxTPR - returning 0.5")
            return 0.5
        idx = int(np.argmax(tpr))
        optimal_threshold = float(threshold[idx])
        logger.debug(f"MaxTPR selected index {idx} with TPR {tpr[idx]:.6f}, threshold {optimal_threshold:.6f}")
        return optimal_threshold


class GeometricRoot(IThresholdOptimizer):
    """
    Chooses threshold that maximizes geometric root tpr * (1 - fpr).
    """

    def calculate_optimal_threshold(self, tpr: ndarray, fpr: ndarray, threshold: ndarray) -> float:
        logger.debug("GeometricRoot.calculate_optimal_threshold started")
        if len(tpr) == 0 or len(fpr) == 0 or len(threshold) == 0:
            logger.warning("Empty arrays provided to GeometricRoot - returning 0.5")
            return 0.5
        scores = tpr * (1 - fpr)
        idx = int(np.argmax(scores))
        optimal_threshold = float(threshold[idx])
        logger.debug(f"GeometricRoot selected index {idx} with score {scores[idx]:.6f}, threshold {optimal_threshold:.6f}")
        return optimal_threshold
