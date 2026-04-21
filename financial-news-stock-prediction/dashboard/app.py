import datetime as dt
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_collection import (
    download_stock_data,
    get_default_date_range,
    fetch_real_news,
)
from src.feature_engineering import (
    build_supervised_dataset,
    compute_price_features,
    merge_price_and_sentiment,
)
from src.lstm_model import predict_next_movement, train_lstm_on_dataframe
from src.sentiment_model import FinBertSentimentAnalyzer, aggregate_daily_sentiment

import sys


def normalize_price_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})

    return df


def run_dashboard() -> None:
    st.set_page_config(
        page_title="Financial AI Predictor",
        layout="wide",
    )

    st.title("📈 Stock Prediction using News Sentiment")

    # Sidebar
    ticker = st.sidebar.selectbox("Ticker", ["AAPL", "TSLA", "AMZN"])

    default_start, default_end = get_default_date_range(days_back=90)

    start_date = st.sidebar.date_input("Start", dt.date.fromisoformat(default_start))
    end_date = st.sidebar.date_input("End", dt.date.fromisoformat(default_end))

    if start_date >= end_date:
        st.sidebar.error("Invalid date range")
        return

    if not st.sidebar.button("Run"):
        return

    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    # =========================
    # STOCK DATA
    # =========================
    prices = download_stock_data(ticker, start_str, end_str)
    prices = normalize_price_dataframe(prices)

    st.subheader("Stock Price")
    st.dataframe(prices.tail())

    fig = px.line(prices, x="date", y="Close", title="Price")
    st.plotly_chart(fig, width="stretch")

    # =========================
    # NEWS (NO API 🔥)
    # =========================
    news_df = fetch_real_news(ticker)

    if news_df.empty:
        st.warning("No news found")
        daily_sentiment = pd.DataFrame({
            "date": pd.to_datetime(prices["date"]).dt.date,
            "daily_sentiment": 0.0,
            "num_headlines": 0,
        }).drop_duplicates(subset=["date"])
    else:
        analyzer = FinBertSentimentAnalyzer()
        news_scored = analyzer.score_dataframe(news_df, text_column="headline")
        daily_sentiment = aggregate_daily_sentiment(news_scored)

    # =========================
    # SENTIMENT GRAPH
    # =========================
    st.subheader("Sentiment")

    fig_sent = px.bar(
        daily_sentiment,
        x="date",
        y="daily_sentiment",
        title="Daily Sentiment",
    )
    st.plotly_chart(fig_sent, width="stretch")

    # =========================
    # HEADLINES WITH SENTIMENT
    # =========================
    st.subheader("📰 Latest News with Sentiment")

    news_scored = pd.DataFrame()
    if not news_df.empty:
        analyzer = FinBertSentimentAnalyzer()
        news_scored = analyzer.score_dataframe(news_df, text_column="headline")
        daily_sentiment = aggregate_daily_sentiment(news_scored)  # Recompute if needed
        
        # Show detailed news with sentiment
        st.dataframe(
            news_scored[['date', 'headline', 'sentiment_label', 'sentiment_numeric']].head(10),
            use_container_width=True,
            hide_index=True
        )
        
        # Sentiment distribution chart
        if len(news_scored) > 0:
            fig_news_sent = px.histogram(
                news_scored.head(20),
                x="sentiment_numeric",
                color="sentiment_label",
                title="News Sentiment Distribution",
                nbins=20
            )
            st.plotly_chart(fig_news_sent, use_container_width=True)
    else:
        st.info("No news data available")


    # =========================
    # MODEL
    # =========================
    price_features = compute_price_features(prices, ma_window=5)
    merged = merge_price_and_sentiment(price_features, daily_sentiment)

    feature_cols = ["daily_return", "ma_close", "Volume", "daily_sentiment"]

    supervised = build_supervised_dataset(merged, feature_columns=feature_cols)

    if len(supervised) < 20:
        st.warning("Increase date range")

    training_result = train_lstm_on_dataframe(
        supervised,
        feature_columns=feature_cols,
        seq_len=5,
        epochs=5,
    )

    pred_label, prob_up = predict_next_movement(training_result, supervised)

    # =========================
    # RESULT
    # =========================
    st.subheader("Prediction")

    direction = "📈 UP" if pred_label == 1 else "📉 DOWN"

    st.metric("Next Day", direction, f"{prob_up:.2%}")


if __name__ == "__main__":
    run_dashboard()
