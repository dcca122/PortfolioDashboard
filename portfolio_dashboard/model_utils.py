from __future__ import annotations

"""Helper functions for model training and signal generation."""

from typing import Iterable, Tuple
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from .config import XGB_PARAMS, SIGNAL_LOWER_Q, SIGNAL_UPPER_Q


def create_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Return future percentage return for each stock."""
    return (
        df["Close"].groupby(level="Ticker").pct_change(horizon).shift(-horizon)
    )


def prepare_features(df: pd.DataFrame, horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    """Create training features ``X`` and binary target ``y``."""
    target = create_target(df, horizon)
    feature_cols = [c for c in df.columns if c != "Close"]
    X = df[feature_cols].copy()
    X = X.loc[X.index.intersection(target.dropna().index)]
    y = (target.loc[X.index] > 0).astype(int)
    return X, y


def train_xgb_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    n_splits: int = 3,
) -> XGBClassifier:
    """Train an XGBoost classifier using expanding-window CV."""
    params = params or XGB_PARAMS
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_model: XGBClassifier | None = None
    best_score = -np.inf
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        score = model.best_score if hasattr(model, "best_score") else model.score(X_val, y_val)
        if score > best_score:
            best_score = score
            best_model = model
    assert best_model is not None
    best_model.fit(X, y)
    return best_model


def save_model(model: XGBClassifier, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> XGBClassifier:
    with open(path, "rb") as f:
        return pickle.load(f)


def generate_signals(
    probs: pd.Series,
    lower_q: float = SIGNAL_LOWER_Q,
    upper_q: float = SIGNAL_UPPER_Q,
) -> pd.DataFrame:
    """Return DataFrame with probability and long/short signal."""
    signals = pd.DataFrame({"prob_up": probs})
    lower = probs.quantile(lower_q)
    upper = probs.quantile(upper_q)
    signals["signal"] = 0
    signals.loc[probs >= upper, "signal"] = 1
    signals.loc[probs <= lower, "signal"] = -1
    return signals


def compute_shap_values(model: XGBClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """Compute SHAP values for interpretability (optional)."""
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - missing optional dep
        raise ImportError("shap is required to compute SHAP values") from exc
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    return pd.DataFrame(values, index=X.index, columns=X.columns)
