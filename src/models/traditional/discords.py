import numpy as np
import stumpy # Matrix Profile library
from src.models.base import Detector

class MatrixProfileDiscordDetector(Detector):
    """
    Anomaly detector based on Time Series Discords using the Matrix Profile.
    Higher matrix profile values (distances) indicate more likely anomalies.
    """
    def __init__(self, window_size: int = 200):
        super().__init__()
        self.window_size = window_size

    def fit(self, x_train: np.ndarray):
        """
        Fit the detector. For this discord detector, fit might not do much.
        It could be used to validate data or for more advanced strategies.
        """
        if x_train.ndim != 1:
            raise ValueError("Input x_train must be a 1D NumPy array.")
        pass

    def score(self, x_test: np.ndarray, window_size: int = None) -> np.ndarray:
        """
        Compute anomaly scores using the Matrix Profile on x_test.
        The score for a point i is the matrix profile value for the window starting at i.

        Args:
            x_test (np.ndarray): The test data (1D).
            window_size (int, optional): The size of the subsequence window for this specific dataset.
                                         If None, uses self.window_size.

        Returns:
            np.ndarray: Anomaly scores for each data point in x_test.
        """
        if window_size is None:
            window_size = self.window_size

        if not isinstance(window_size, (np.int64, int)):
            raise TypeError(f"window_size must be an integer. Got {type(window_size)}")
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if x_test.ndim != 1:
            raise ValueError("Input x_test must be a 1D NumPy array.")

        if len(x_test) < window_size:
            print(f"Warning: Length of x_test ({len(x_test)}) is less than dynamic window_size ({window_size}). "
                  "Returning zero scores.")
            return np.zeros_like(x_test, dtype=float)

        try:
            # Ensure x_test is float64 for stumpy, and not readonly
            x_test_stumpy = np.ascontiguousarray(x_test, dtype=np.float64)
            mp_values = stumpy.stump(x_test_stumpy, m=window_size)[:, 0]
        except Exception as e:
            print(f"Error during stumpy.stump computation with window_size {window_size}: {e}. Returning zero scores.")
            return np.zeros_like(x_test, dtype=float)

        scores = np.zeros(len(x_test), dtype=float)
        
        if len(mp_values) > 0:
            scores[:len(mp_values)] = mp_values
            padding_value = np.mean(mp_values)
            scores[len(mp_values):] = padding_value
        
        return scores
