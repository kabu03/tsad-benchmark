from abc import ABC, abstractmethod # ABC = Abstract Base Class
import numpy as np

class Detector(ABC):
    """
    Abstract base class for anomaly detectors.
    """

    @abstractmethod
    def fit(self, x_train: np.ndarray):
        """
        Fit the detector to the training data.

        Args:
            x_train (np.ndarray): The training data.
        """
        pass

    @abstractmethod
    def score(self, x_test: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for the test data.

        Args:
            x_test (np.ndarray): The test data.

        Returns:
            np.ndarray: Anomaly scores for each data point in x_test.
        """
        pass
