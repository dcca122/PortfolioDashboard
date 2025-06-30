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


