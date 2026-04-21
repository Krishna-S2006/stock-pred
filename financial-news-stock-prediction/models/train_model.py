from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.data_collection import download_stock_data
from src.feature_engineering import (
    build_supervised_dataset,
    compute_price_features,
    merge_price_and_sentiment,
)
from src.lstm_model import predict_next_movement, train_lstm_on_dataframe
from src.preprocessing import filter_news, load_sample_news
from src.sentiment_model import FinBertSentimentAnalyzer, aggregate_daily_sentiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_pipeline(ticker: str, start: str, end: str) -> None:
    print(f"Running pipeline for {ticker} from {start} to {end}")

    # 1. Stock data
    prices = download_stock_data(ticker, start, end)
    print(f"Downloaded {len(prices)} rows of price data.")

    # 2. News and sentiment
    news = load_sample_news()
    news_filtered = filter_news(news, ticker=ticker, start=start, end=end)
    print(f"Loaded {len(news_filtered)} news headlines for sentiment analysis.")

    if news_filtered.empty:
        print("No news available for this period. Proceeding with neutral sentiment.")
        daily_sentiment = pd.DataFrame(
            {"date": pd.to_datetime(prices["date"]).dt.date, "daily_sentiment": 0.0, "num_headlines": 0}
        ).drop_duplicates(subset=["date"])
    else:
        analyzer = FinBertSentimentAnalyzer()
        news_scored = analyzer.score_dataframe(news_filtered, text_column="headline")
        daily_sentiment = aggregate_daily_sentiment(news_scored)

    # 3. Feature engineering
    price_features = compute_price_features(prices, ma_window=5)
    merged = merge_price_and_sentiment(price_features, daily_sentiment)

    feature_cols = ["daily_return", "ma_close", "Volume", "daily_sentiment"]
    supervised = build_supervised_dataset(merged, feature_columns=feature_cols)
    print(f"Supervised dataset rows: {len(supervised)}")

    if len(supervised) < 20:
        print("Warning: very few samples available for training. Results may be unstable.")

    # 4. Train LSTM
    training_result = train_lstm_on_dataframe(
        supervised,
        feature_columns=feature_cols,
        target_column="target_up",
        seq_len=5,
        epochs=10,
        lr=1e-3,
        batch_size=16,
    )

    # 5. Predict next movement
    pred_label, prob_up = predict_next_movement(training_result, supervised)
    direction = "UP" if pred_label == 1 else "DOWN"
    print(f"Predicted next-day movement: {direction} (prob_up={prob_up:.3f})")

    # Example of saving model (optional)
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True, parents=True)

    model_path = models_dir / f"lstm_{ticker}.pt"
    torch.save(training_result.model.state_dict(), model_path)
    print(f"Saved trained model weights to: {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM model for stock movement prediction.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol.")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", type=str, default="2023-03-31", help="End date YYYY-MM-DD.")

    args = parser.parse_args()
    run_pipeline(args.ticker.upper(), args.start, args.end)


if __name__ == "__main__":
    main()

