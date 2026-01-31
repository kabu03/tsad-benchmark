import numpy as np
from typing import Any
from sklearn.ensemble import IsolationForest
from src.models.base import Detector

class IsolationForestDetector(Detector):
    """
    Anomaly detector using Isolation Forest.
    """
    def __init__(self, n_estimators: int = 100, contamination: Any = 'auto', random_state: int = 42, **kwargs):
        """
        Args:
            n_estimators (int): Number of base estimators in the ensemble.
            contamination (str or float): The amount of contamination of the data set, i.e.,
                                          the proportion of outliers in the data set. Used when
                                          fitting to define the threshold on the scores.
                                          If 'auto', the threshold is determined as in the original paper.
                                          For our use case where predict takes an explicit threshold,
                                          this primarily affects the `predict` method of the sklearn model,
                                          not `decision_function` directly for thresholding.
            random_state (int): Controls the pseudo-randomness of the selection of the feature
                                and split values for each branching.
            **kwargs: Other keyword arguments for sklearn.ensemble.IsolationForest.
                      Example: max_samples, max_features, bootstrap, n_jobs, warm_start.
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = IsolationForest(n_estimators=self.n_estimators,
                                     contamination=self.contamination,
                                     random_state=self.random_state,
                                     **self.kwargs)

    def fit(self, x_train: np.ndarray):
        """
        Fit the Isolation Forest model to the training data.
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
        self.model.fit(x_train_reshaped)

    def score(self, x_test: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for the test data.
        Scores are inverted decision_function values (higher = more anomalous).
        The decision_function of IsolationForest returns higher scores for inliers
        and lower scores for outliers.

        Args:
            x_test (np.ndarray): The test data (1D).
                                 It will be reshaped to (n_samples, 1).

        Returns:
            np.ndarray: Anomaly scores for each data point in x_test.
        """
        if not hasattr(self.model, 'estimators_'):
             raise RuntimeError("The detector has not been fitted yet. Call fit() first.")

        if x_test.ndim != 1:
            if x_test.ndim == 2 and x_test.shape[1] == 1:
                pass
            else:
                raise ValueError("Input x_test must be a 1D NumPy array or a 2D array with 1 feature.")

        x_test_reshaped = x_test.reshape(-1, 1) if x_test.ndim == 1 else x_test
        
        # decision_function: The lower, the more abnormal. Negative scores are outliers, positive are inliers.
        # We invert it so that higher scores are more anomalous.
        return -self.model.decision_function(x_test_reshaped)
