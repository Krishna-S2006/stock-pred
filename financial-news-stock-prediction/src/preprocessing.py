from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_sample_news(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the example news dataset from CSV.

    Parameters
    ----------
    path : str, optional
        Optional custom path to the CSV. If None, uses data/sample_news.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, ticker, headline, source.
    """
    if path is None:
        path = DATA_DIR / "sample_news.csv"

    df = pd.read_csv(path)
    if "date" not in df.columns or "headline" not in df.columns:
        raise ValueError("sample_news.csv must contain at least 'date' and 'headline' columns")

    df["date"] = pd.to_datetime(df["date"]).dt.date
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


def filter_news(
    news_df: pd.DataFrame,
    ticker: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter news by ticker and date range.

    Parameters
    ----------
    news_df : pd.DataFrame
        Input news DataFrame.
    ticker : str, optional
        Ticker symbol to filter (if present in the data).
    start : str, optional
        Start date in YYYY-MM-DD.
    end : str, optional
        End date in YYYY-MM-DD.
    """
    df = news_df.copy()

    if ticker is not None and "ticker" in df.columns:
        df = df[df["ticker"].str.upper() == ticker.upper()]

    if start is not None:
        start_d = pd.to_datetime(start).date()
        df = df[df["date"] >= start_d]

    if end is not None:
        end_d = pd.to_datetime(end).date()
        df = df[df["date"] <= end_d]

    return df.reset_index(drop=True)


__all__ = ["load_sample_news", "filter_news"]

