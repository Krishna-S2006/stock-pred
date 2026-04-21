from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def compute_price_features(
    price_df: pd.DataFrame,
    ma_window: int = 5,
) -> pd.DataFrame:
    """
    Compute price features from stock dataframe returned by yfinance.
    Handles MultiIndex columns and ensures correct data types.
    """

    df = price_df.copy()

    # If Date is index -> convert to column
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # Handle MultiIndex columns (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Normalize column names
    df.columns = [str(c).lower() for c in df.columns]

    # Rename columns
    rename_map = {
        "date": "date",
        "close": "Close",
        "volume": "Volume"
    }

    df = df.rename(columns=rename_map)

    # Check required columns
    required_cols = {"date", "Close", "Volume"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Price DataFrame missing columns: {missing}")

    # Convert types
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df = df.sort_values("date")

    # Feature engineering
    df["daily_return"] = df["Close"].pct_change()
    df["ma_close"] = df["Close"].rolling(window=ma_window, min_periods=1).mean()

    return df

def merge_price_and_sentiment(
    price_features: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge price features and daily sentiment on date.
    """

    for col in ("date", "daily_sentiment"):
        if col not in daily_sentiment.columns:
            raise ValueError(f"Sentiment DataFrame missing column '{col}'")

    df_price = price_features.copy()
    df_sent = daily_sentiment.copy()

    df_sent["date"] = pd.to_datetime(df_sent["date"]).dt.date
    df_price["date"] = pd.to_datetime(df_price["date"]).dt.date

    merged = pd.merge(df_price, df_sent, on="date", how="left")

    merged["daily_sentiment"] = merged["daily_sentiment"].fillna(0.0)
    merged["num_headlines"] = merged.get("num_headlines", 0).fillna(0)

    return merged


def build_supervised_dataset(
    merged_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Create dataset with binary target:
    1 -> next day price increases
    0 -> next day price decreases
    """

    df = merged_df.copy().sort_values("date").reset_index(drop=True)

    if "Close" not in df.columns:
        raise ValueError("Column 'Close' is required in merged_df")

    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' not in merged_df")

    df["next_close"] = df["Close"].shift(-1)

    df["target_up"] = (df["next_close"] > df["Close"]).astype(int)

    df = df.dropna(subset=["next_close"])

    return df


__all__ = [
    "compute_price_features",
    "merge_price_and_sentiment",
    "build_supervised_dataset",
]


if __name__ == "__main__":

    print("Loading stock data...")

    price_df = pd.read_csv("data/stock_data.csv")

    print("Computing price features...")

    price_features = compute_price_features(price_df)

    # temporary demo sentiment
    sentiment_df = pd.DataFrame({
        "date": price_features["date"].head(10),
        "daily_sentiment": np.random.uniform(-1, 1, 10),
        "num_headlines": np.random.randint(1, 10, 10)
    })

    print("Merging price and sentiment...")

    merged = merge_price_and_sentiment(price_features, sentiment_df)

    print("Building supervised dataset...")

    dataset = build_supervised_dataset(
        merged,
        feature_columns=[
            "daily_return",
            "ma_close",
            "Volume",
            "daily_sentiment"
        ],
    )

    dataset.to_csv("data/final_dataset.csv", index=False)

    print("Final dataset saved to data/final_dataset.csv")
    print(dataset.head())
