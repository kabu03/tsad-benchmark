import numpy as np
from src.models.base import Detector

class ZScoreDetector(Detector):
    """
    Anomaly detector based on Z-scores.
    Uses mean and standard deviation from training data to calculate Z-scores for test data.
    """
    def __init__(self):
        super().__init__()
        self.mean_train = None
        self.std_train = None

    def fit(self, x_train: np.ndarray):
        """
        Calculate mean and standard deviation from the training data.

        Args:
            x_train (np.ndarray): The training data (1D).
        """
        if x_train.ndim != 1:
            raise ValueError("Input x_train must be a 1D NumPy array.")
        if len(x_train) == 0:
            raise ValueError("Input x_train cannot be empty.")

        self.mean_train = np.mean(x_train)
        self.std_train = np.std(x_train)

        if self.std_train == 0:
            print("Warning: Standard deviation of training data is zero. Z-scores will be NaN or Inf for differing points.")

    def score(self, x_test: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores (absolute Z-scores) for the test data.

        Args:
            x_test (np.ndarray): The test data (1D).

        Returns:
            np.ndarray: Absolute Z-scores for each data point in x_test.
                        Returns np.nan for points where std_train is 0 and point differs from mean.
                        Returns 0 for points where std_train is 0 and point equals mean.
        """
        if self.mean_train is None or self.std_train is None:
            raise RuntimeError("The detector has not been fitted yet. Call fit() first.")
        if x_test.ndim != 1:
            raise ValueError("Input x_test must be a 1D NumPy array.")

        if self.std_train == 0:
            scores = np.abs((x_test - self.mean_train) / (self.std_train + 1e-10)) # Add epsilon to avoid direct 0/0 if x_test == mean
        else:
            scores = np.abs((x_test - self.mean_train) / self.std_train)
        
        return scores
