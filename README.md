# PortfolioDashboard

This repository contains utilities to build a feature dataset for
stock modelling experiments. The code retrieves weekly price data,
calculates momentum factors, constructs a simple stock network and
exports the merged features for use in machine learning models.

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline for a set of sample tickers:

```bash
python -m portfolio_dashboard.pipeline
```

This downloads weekly prices from Yahoo! Finance and writes a table of
lagged factors with graph metrics.



## Model training

After generating the feature dataset you can train the XGBoost model and generate trading signals:

```bash
python -m portfolio_dashboard.train_model --data features.parquet \
    --model-out xgb_model.pkl --pred-out predictions.csv
```

Optional SHAP values can be saved with `--shap-out shap.csv`.
