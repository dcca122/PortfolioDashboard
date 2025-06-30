from __future__ import annotations

"""Command line utility to train the XGBoost model and generate signals."""

import argparse
import pandas as pd

from .model_utils import (
    prepare_features,
    train_xgb_classifier,
    save_model,
    generate_signals,
    compute_shap_values,
)


def load_dataset(path: str) -> pd.DataFrame:
    """Load features dataset from CSV or Parquet."""
    if path.endswith(".csv"):
        df = pd.read_csv(path, parse_dates=["Date"], index_col=["Date", "Ticker"])
    else:
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.MultiIndex):
            df.set_index(["Date", "Ticker"], inplace=True)
    return df


def main(args: argparse.Namespace) -> None:
    df = load_dataset(args.data)
    X, y = prepare_features(df)

    model = train_xgb_classifier(X, y)

    probs = pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="prob_up")
    signals = generate_signals(probs)
    signals.to_csv(args.pred_out)

    save_model(model, args.model_out)

    if args.shap_out:
        shap_df = compute_shap_values(model, X)
        shap_df.to_csv(args.shap_out)


if __name__ == "__main__":  # pragma: no cover - manual execution
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--data", required=True, help="Path to feature dataset")
    parser.add_argument(
        "--model-out", default="xgb_model.pkl", help="File to save trained model"
    )
    parser.add_argument(
        "--pred-out", default="predictions.csv", help="File to save predictions"
    )
    parser.add_argument(
        "--shap-out",
        default=None,
        help="Optional path to write shap values as CSV",
    )
    main(parser.parse_args())
