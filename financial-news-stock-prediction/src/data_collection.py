import datetime as dt
from typing import Optional

import pandas as pd
import yfinance as yf


# =========================
# STOCK DATA
# =========================
def download_stock_data(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:

    df = yf.download(ticker, start=start, end=end, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    df = df.reset_index()

    # Fix multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date

    return df


# =========================
# DATE RANGE
# =========================
def get_default_date_range(
    days_back: int = 90,
    end: Optional[dt.date] = None,
) -> tuple[str, str]:

    if end is None:
        end = dt.date.today()

    start = end - dt.timedelta(days=days_back)

    return start.isoformat(), end.isoformat()


# =========================
# NEWS (NO API KEY 🔥)
# =========================
def fetch_real_news(ticker: str) -> pd.DataFrame:
    """
    Fetch latest news using yfinance (no API key required)
    """

    stock = yf.Ticker(ticker)
    news = stock.news

    # Debug: Check if news is None or empty
    if news is None:
        print(f"Warning: No news data returned for {ticker}")
        return pd.DataFrame(columns=["date", "headline"])

    if not news:
        print(f"Warning: Empty news list for {ticker}")
        return pd.DataFrame(columns=["date", "headline"])

    data = []
    for article in news:
        try:
            # Updated: Use pubDate instead of providerPublishTime
            pub_date = article.get("content", {}).get("pubDate")
            
            if pub_date:
                # Parse ISO format date string
                date_obj = pd.to_datetime(pub_date).date()
            else:
                print(f"Warning: No pubDate found in article")
                continue
            
            title = article.get("content", {}).get("title")
            if not title:
                print(f"Warning: No title found in article")
                continue
                
            data.append({
                "date": date_obj,
                "headline": title,
            })
        except Exception as e:
            print(f"Error processing article: {e}")
            continue

    if not data:
        print(f"Warning: No valid articles extracted for {ticker}")
        return pd.DataFrame(columns=["date", "headline"])

    df = pd.DataFrame(data)
    return df
