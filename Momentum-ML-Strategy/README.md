# Momentum ML Strategy

This project demonstrates how to combine classic momentum indicators with
machine learning models to trade a basket of equities. The workflow mirrors
a realistic research pipeline:

1. **Data ingestion** – Download daily close prices for a user-specified list
   of tickers (defaults to large-cap technology names) from Yahoo Finance via
   [`yfinance`](https://github.com/ranaroussi/yfinance).
2. **Feature engineering** – Build multiple momentum features (different lookback
   horizons, moving-average ratios, volatility measures, and price position
   within a rolling window) plus the Hurst exponent to detect persistent trends.
3. **Model training** – Fit a linear regression model and an extra-trees
   regressor with chronological train/validation/test splits to avoid leakage.
4. **Backtesting** – Generate long/short signals conditioned on the Hurst
   threshold (> 0.5) and simulate portfolio performance with an equal-weighted
   allocation across active positions.
5. **Diagnostics** – Persist per-ticker signal plots (green markers for long,
   red markers for short), CSV exports for further analysis, and JSON model
   diagnostics.

## Key files

- `momentum_ml_strategy.py`: End-to-end script that downloads the data, trains
  the models, runs the backtest, and writes artefacts to `results/`.
- `results/`: Created automatically when the script runs. Contains portfolio
  equity curve, per-trade details, and signal plots.

## Usage

```bash
python momentum_ml_strategy.py --start 2023-06-01 --end 2025-10-30 \
  --tickers AAPL MSFT GOOG AMZN NVDA META BRK-B ORCL AMD TSLA --capital 150000
```

Omit `--end` to default to the latest available date. The script automatically
splits the data into 60% training, 20% validation, and 20% test segments, which
are used respectively for model fitting, hyper-parameter selection, and
out-of-sample evaluation.

## Dependencies

- Python 3.10+
- `pandas`, `numpy`, `yfinance`
- `scikit-learn`
- `matplotlib`
- Optional: `auquan-toolbox` (for the Hurst exponent helper). The script ships
  with a fallback implementation so it works even if the toolbox cannot be
  installed in your environment.

> **Note:** In network-restricted environments `pip install auquan-toolbox` may
> fail. The fallback keeps the strategy runnable, but installing the official
> toolbox is recommended for parity with Auquan's implementation.

## Outputs

After a successful run, inspect the `results/` directory for:

- `model_summary.json` – training/validation metrics for each model
- `by_trade.csv` – detailed per-ticker signal and PnL history
- `daily_returns.csv` – aggregated daily returns and net exposures
- `portfolio_value.csv` – equity curve of the strategy starting from the
  configured initial capital
- `signals_<TICKER>.png` – scatter plot of long (green) and short (red) markers
  overlayed on the closing price for each ticker.

These artefacts make it easy to iterate on the feature set, swap in additional
models, or plug the signals into a live execution engine.
