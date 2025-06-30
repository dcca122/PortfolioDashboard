from __future__ import annotations

"""Configuration constants for model training."""

XGB_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
}

SIGNAL_UPPER_Q = 0.9
SIGNAL_LOWER_Q = 0.1
