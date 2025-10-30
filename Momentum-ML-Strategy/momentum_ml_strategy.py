"""Momentum-based multi-asset trading strategy powered by machine learning.

This module downloads historical equity data, engineers momentum features,
trains a couple of regressors (linear regression and extra trees), and
backtests a simple long/short strategy that only trades when the Hurst
exponent suggests persistent price behaviour (H > 0.5).

The code is organised so it can be imported as a module or executed as a
script. When executed directly it will download the data, fit the models,
run the backtest, and save diagnostic artefacts (plots, CSV files) inside a
``results`` folder.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import yfinance as yf

try:  # pragma: no cover - optional dependency
    from auquan_toolbox.core.utilities import hurst as auquan_hurst  # type: ignore
except Exception:  # pragma: no cover - auquan toolbox not available in CI

    def auquan_hurst(series: Iterable[float]) -> float:
        """Fallback Hurst exponent implementation.

        Uses a simple rescaled range approach. This is less feature-rich than
        the auquan version but works without the optional dependency.
        """

        series = np.asarray(list(series), dtype=float)
        if series.size < 20:
            return np.nan
        lags = range(2, min(100, series.size // 2))
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        slope, _ = np.polyfit(np.log(lags), np.log(tau), 1)
        return slope * 2.0


@dataclass
class StrategyConfig:
    tickers: Tuple[str, ...]
    start_date: str
    end_date: Optional[str] = None
    initial_capital: float = 100_000.0
    hurst_window: int = 100
    hurst_threshold: float = 0.5
    train_split: float = 0.6
    validation_split: float = 0.2
    prediction_horizon: int = 1  # days ahead for the target variable
    results_dir: pathlib.Path = pathlib.Path("results")

    def __post_init__(self) -> None:
        if not math.isclose(self.train_split + self.validation_split, 0.8, abs_tol=1e-6):
            raise ValueError("Train and validation splits must sum to 0.8 (80%).")
        self.results_dir.mkdir(parents=True, exist_ok=True)


def fetch_adjusted_close(tickers: Iterable[str], start_date: str, end_date: Optional[str]) -> pd.DataFrame:
    """Download adjusted close prices using yfinance."""

    data = yf.download(list(tickers), start=start_date, end=end_date or None, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"] if "Close" in data.columns.levels[0] else data["Adj Close"]
    else:
        prices = data
    prices = prices.dropna(how="all")
    return prices


def compute_momentum_features(prices: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Construct a feature matrix with engineered momentum indicators."""

    feature_frames: List[pd.DataFrame] = []
    for ticker in config.tickers:
        series = prices[ticker].dropna()
        df = pd.DataFrame({"close": series})
        df["return_1d"] = df["close"].pct_change()
        df["log_return_1d"] = np.log(df["close"]).diff()

        for window in (5, 10, 21, 63):
            df[f"momentum_{window}"] = df["close"].pct_change(window)
            df[f"sma_{window}"] = df["close"].rolling(window).mean()
            df[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()
            df[f"volatility_{window}"] = df["return_1d"].rolling(window).std()

        df["sma_ratio_5_21"] = df["sma_5"] / df["sma_21"] - 1
        df["sma_ratio_21_63"] = df["sma_21"] / df["sma_63"] - 1
        df["ema_ratio_5_21"] = df["ema_5"] / df["ema_21"] - 1
        df["ema_ratio_21_63"] = df["ema_21"] / df["ema_63"] - 1

        df["rolling_max_21"] = df["close"].rolling(21).max()
        df["rolling_min_21"] = df["close"].rolling(21).min()
        df["price_position_21"] = (df["close"] - df["rolling_min_21"]) / (
            df["rolling_max_21"] - df["rolling_min_21"]
        )

        df["future_return"] = df["close"].pct_change(config.prediction_horizon).shift(-config.prediction_horizon)
        df["future_return"] = df["future_return"].replace([np.inf, -np.inf], np.nan)

        hurst_values = (
            df["close"].rolling(config.hurst_window).apply(lambda x: auquan_hurst(np.log(x + 1e-9)), raw=False)
        )
        df["hurst"] = hurst_values

        df["ticker"] = ticker
        feature_frames.append(df)

    features = pd.concat(feature_frames)
    features.index.name = "date"
    features = features.reset_index().set_index(["date", "ticker"]).sort_index()
    return features


def split_datasets(features: pd.DataFrame, config: StrategyConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/validation/test sets by chronology."""

    dataset = features.dropna(subset=["future_return"]).copy()
    dataset = dataset.dropna()

    unique_dates = dataset.index.get_level_values("date").unique().sort_values()
    train_cutoff = unique_dates[int(len(unique_dates) * config.train_split)]
    val_cutoff = unique_dates[int(len(unique_dates) * (config.train_split + config.validation_split))]

    train = dataset.loc[pd.IndexSlice[:train_cutoff], :]
    validation = dataset.loc[pd.IndexSlice[train_cutoff:val_cutoff], :]
    test = dataset.loc[pd.IndexSlice[val_cutoff:, :], :]

    return train, validation, test


def build_model_pipeline(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def train_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> Dict[str, Dict[str, object]]:
    """Train models and evaluate them on training and validation sets."""

    feature_cols = [col for col in train.columns if col not in {"future_return"}]

    X_train = train[feature_cols].values
    y_train = train["future_return"].values

    X_val = validation[feature_cols].values
    y_val = validation["future_return"].values

    models = {
        "linear_regression": build_model_pipeline(LinearRegression()),
        "extra_trees": build_model_pipeline(ExtraTreesRegressor(
            n_estimators=400,
            max_depth=6,
            random_state=42,
            min_samples_leaf=5,
            n_jobs=-1,
        )),
    }

    results: Dict[str, Dict[str, object]] = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        train_pred = pipeline.predict(X_train)
        val_pred = pipeline.predict(X_val)
        results[name] = {
            "pipeline": pipeline,
            "train_mae": float(mean_absolute_error(y_train, train_pred)),
            "train_rmse": float(math.sqrt(mean_squared_error(y_train, train_pred))),
            "val_mae": float(mean_absolute_error(y_val, val_pred)),
            "val_rmse": float(math.sqrt(mean_squared_error(y_val, val_pred))),
        }

    return results


def pick_best_model(results: Dict[str, Dict[str, object]]) -> str:
    """Choose the model with the lowest validation RMSE."""

    best_model = min(results.items(), key=lambda item: item[1]["val_rmse"])
    return best_model[0]


def backtest_strategy(
    dataset: pd.DataFrame,
    pipelines: Dict[str, Dict[str, object]],
    best_model_name: str,
    config: StrategyConfig,
) -> Dict[str, pd.DataFrame]:
    """Run the backtest on the hold-out test set."""

    feature_cols = [col for col in dataset.columns if col not in {"future_return"}]

    X = dataset[feature_cols].values
    y = dataset["future_return"].values

    preds: Dict[str, np.ndarray] = {}
    for name, info in pipelines.items():
        preds[name] = info["pipeline"].predict(X)

    dataset = dataset.copy()
    for name, pred in preds.items():
        dataset[f"prediction_{name}"] = pred

    dataset["prediction_ensemble"] = dataset[[f"prediction_{name}" for name in preds]].mean(axis=1)
    active_prediction = dataset[f"prediction_{best_model_name}"]
    dataset["prediction_active"] = 0.5 * active_prediction + 0.5 * dataset["prediction_ensemble"]

    dataset["signal"] = 0
    dataset.loc[dataset["hurst"] > config.hurst_threshold, "signal"] = np.sign(dataset.loc[dataset["hurst"] > config.hurst_threshold, "prediction_active"])
    dataset["signal"] = dataset["signal"].replace(-0.0, 0)

    dataset["strategy_return"] = dataset["signal"] * y

    daily = (
        dataset
        .reset_index()
        .groupby("date")
        .agg({
            "strategy_return": "mean",
            "future_return": "mean",
            "signal": "sum",
        })
        .rename(columns={"future_return": "mean_forward_return", "signal": "net_exposure"})
    )

    portfolio = []
    capital = config.initial_capital
    for date, row in daily.iterrows():
        daily_return = row["strategy_return"]
        capital = capital * (1 + daily_return)
        portfolio.append({
            "date": date,
            "daily_return": daily_return,
            "portfolio_value": capital,
            "net_exposure": row["net_exposure"],
        })

    portfolio_df = pd.DataFrame(portfolio).set_index("date")
    dataset = dataset.reset_index()
    dataset["cumulative_pnl"] = dataset.groupby("ticker")["strategy_return"].cumsum()

    return {
        "by_trade": dataset,
        "daily": daily,
        "portfolio": portfolio_df,
    }


def save_results(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    model_results: Dict[str, Dict[str, object]],
    backtest_results: Dict[str, pd.DataFrame],
    config: StrategyConfig,
) -> None:
    """Persist artefacts to the results directory."""

    summary_path = config.results_dir / "model_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(model_results, fh, indent=2)

    backtest_results["by_trade"].to_csv(config.results_dir / "by_trade.csv", index=False)
    backtest_results["daily"].to_csv(config.results_dir / "daily_returns.csv")
    backtest_results["portfolio"].to_csv(config.results_dir / "portfolio_value.csv")

    for ticker, ticker_df in backtest_results["by_trade"].groupby("ticker"):
        plot_signals(ticker_df, ticker, config)


def plot_signals(ticker_df: pd.DataFrame, ticker: str, config: StrategyConfig) -> None:
    """Plot price series with buy/sell markers."""

    price = ticker_df.set_index("date")["close"]
    signals = ticker_df.set_index("date")["signal"]

    longs = signals[signals > 0]
    shorts = signals[signals < 0]

    plt.figure(figsize=(12, 6))
    plt.plot(price.index, price.values, label=f"{ticker} price", color="black")
    plt.scatter(longs.index, price.loc[longs.index], marker="^", color="green", label="Long signal")
    plt.scatter(shorts.index, price.loc[shorts.index], marker="v", color="red", label="Short signal")

    plt.title(f"Signals for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.results_dir / f"signals_{ticker}.png")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Momentum ML strategy backtest")
    parser.add_argument("--start", default="2023-06-01", help="Start date for historical data")
    parser.add_argument("--end", default=None, help="End date (defaults to today)")
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "BRK-B", "ORCL", "AMD", "TSLA"],
        help="List of tickers to include",
    )
    parser.add_argument("--capital", type=float, default=100_000.0, help="Initial portfolio capital")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StrategyConfig(
        tickers=tuple(args.tickers),
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
    )

    prices = fetch_adjusted_close(config.tickers, config.start_date, config.end_date)
    features = compute_momentum_features(prices, config)

    train, validation, test = split_datasets(features, config)
    model_results = train_models(train, validation)

    best_model_name = pick_best_model(model_results)

    backtest_results = backtest_strategy(test, model_results, best_model_name, config)
    save_results(train, validation, test, model_results, backtest_results, config)

    portfolio = backtest_results["portfolio"]
    final_value = portfolio["portfolio_value"].iloc[-1]
    total_return = final_value / config.initial_capital - 1

    print(f"Best model: {best_model_name}")
    print(f"Final portfolio value: {final_value:,.2f}")
    print(f"Total return: {total_return:.2%}")


if __name__ == "__main__":
    main()
