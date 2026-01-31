import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from src.models.base import Detector

class LOFDetector(Detector):
    """
    Anomaly detector using Local Outlier Factor (LOF) with novelty detection.
    """
    def __init__(self, n_neighbors: int = 20, **kwargs):
        """
        Args:
            n_neighbors (int): Number of neighbors to use by default for LOF.
            **kwargs: Other keyword arguments for sklearn.neighbors.LocalOutlierFactor.
                      Example: algorithm, leaf_size, metric, p, metric_params, n_jobs.
                      Note: 'novelty' is fixed to True. 'contamination' is not directly
                      used for thresholding here as an explicit threshold is passed to predict.
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.kwargs = kwargs
        self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True, **self.kwargs)

    def fit(self, x_train: np.ndarray):
        """
        Fit the LOF model to the training data.
        x_train is expected to be normal data.

        Args:
            x_train (np.ndarray): The training data (1D).
                                  It will be reshaped to (n_samples, 1).
        """
        if x_train.ndim != 1:
            if x_train.ndim == 2 and x_train.shape[1] == 1:
                pass
            else:
                raise ValueError("Input x_train must be a 1D NumPy array or a 2D array with 1 feature.")
        
        x_train_reshaped = x_train.reshape(-1, 1) if x_train.ndim == 1 else x_train

        if x_train_reshaped.shape[0] < self.n_neighbors:
            print(f"Warning: Number of training samples ({x_train_reshaped.shape[0]}) is less than n_neighbors ({self.n_neighbors}). "
                  "Adjusting n_neighbors to {x_train_reshaped.shape[0] -1} or model might fail.")
            
            new_n_neighbors = max(1, x_train_reshaped.shape[0] - 1)
            self.model = LocalOutlierFactor(n_neighbors=new_n_neighbors, novelty=True, **self.kwargs)

        self.model.fit(x_train_reshaped)

    def score(self, x_test: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for the test data.
        Scores are inverted decision_function values (higher = more anomalous).

        Args:
            x_test (np.ndarray): The test data (1D).
                                 It will be reshaped to (n_samples, 1).

        Returns:
            np.ndarray: Anomaly scores for each data point in x_test.
        """
        if not hasattr(self.model, 'offset_'): # A simple check if model was fitted
            raise RuntimeError("The detector has not been fitted yet. Call fit() first.")
        
        if x_test.ndim != 1:
            if x_test.ndim == 2 and x_test.shape[1] == 1:
                pass
            else:
                raise ValueError("Input x_test must be a 1D NumPy array or a 2D array with 1 feature.")

        x_test_reshaped = x_test.reshape(-1, 1) if x_test.ndim == 1 else x_test
        
        # decision_function: lower values are more anomalous. We invert it.
        return -self.model.decision_function(x_test_reshaped)
