import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error


def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)**0.5

SCORER = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "mape": mean_absolute_percentage_error,
    "msle": mean_squared_log_error,
    "rmse": root_mean_squared_error
}

def _get_scorer(scoring):
    if scoring not in SCORER:
        scoring = "rmse"
    return scoring, SCORER[scoring]
