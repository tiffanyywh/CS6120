from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from scipy.stats import pearsonr, spearmanr
import numpy as np
from sklearn.metrics import accuracy_score


def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "Pearson_r": r, "Spearman_rho": rho}


def cefr_accuracy_scorer(y_true, y_pred):
    """
    Round continuous CEFR predictions and compute classification accuracy.
    """
    y_true_rounded = np.clip(np.round(y_true), 1, 5)
    y_pred_rounded = np.clip(np.round(y_pred), 1, 5)
    return accuracy_score(y_true_rounded, y_pred_rounded)


def cefr_accuracy_within1(y_true, y_pred):
    y_true_r = np.clip(np.round(y_true), 1, 5)
    y_pred_r = np.clip(np.round(y_pred), 1, 5)
    return np.mean(np.abs(y_true_r - y_pred_r) <= 1)
