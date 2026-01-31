import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def calculate_ranking_metrics(y_true_binary: np.ndarray, y_scores: np.ndarray):
    """
    Calculates threshold-independent metrics: AUC-ROC and PR-AUC.
    """
    if len(np.unique(y_true_binary)) < 2:
        auc_roc = 0.5 
        pr_auc = 0.0
    else:
        try:
            auc_roc = roc_auc_score(y_true_binary, y_scores)
            pr_auc = average_precision_score(y_true_binary, y_scores)
        except Exception:
             auc_roc = 0.5
             pr_auc = 0.0

    return {
        "auc_roc": auc_roc,
        "pr_auc": pr_auc
    }

def calculate_top_k_overlap(anomaly_scores: np.ndarray, true_binary: np.ndarray, k: int = None):
    """
    Checks if the Top-K highest scores overlap with the ground truth.
    If k is None, we assume k = number of true anomalies (Oracle Top-K).
    Returns 1 if ANY of the top-k highest scores fall within a true anomaly region, 0 otherwise.
    """
    if k is None:
        k = int(np.sum(true_binary))
    
    if k == 0:
        return 0

    top_k_indices = np.argsort(anomaly_scores)[-k:]
    
    hits = np.sum(true_binary[top_k_indices])
    
    return 1 if hits > 0 else 0
