from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


FINBERT_MODEL_NAME = "ProsusAI/finbert"


@dataclass
class SentimentResult:
    label: str
    score: float
    numeric_score: float


class FinBertSentimentAnalyzer:
    """
    Wrapper around the FinBERT sentiment analysis pipeline.

    Converts categorical labels (POSITIVE / NEGATIVE / NEUTRAL)
    into a numeric sentiment score:
        POSITIVE -> +score
        NEGATIVE -> -score
        NEUTRAL  -> 0
    """

    def __init__(self, model_name: str = FINBERT_MODEL_NAME, device: int = -1) -> None:
        # Use AutoModel/Tokenizer explicitly so the first load is predictable,
        # then wrap with transformers.pipeline for convenience.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

    @staticmethod
    def _to_numeric(label: str, score: float) -> float:
        label = label.upper()
        if "POSITIVE" in label:
            return float(score)
        if "NEGATIVE" in label:
            return float(-score)
        return 0.0

    def analyze_texts(self, texts: Iterable[str]) -> List[SentimentResult]:
        outputs = self._pipeline(list(texts), truncation=True)
        results: List[SentimentResult] = []
        for out in outputs:
            label = out["label"]
            score = float(out["score"])
            numeric = self._to_numeric(label, score)
            results.append(SentimentResult(label=label, score=score, numeric_score=numeric))
        return results

    def score_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "headline",
    ) -> pd.DataFrame:
        """
        Add sentiment columns to a DataFrame of news.

        Adds:
        - sentiment_label
        - sentiment_score
        - sentiment_numeric
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        texts = df[text_column].astype(str).tolist()
        results = self.analyze_texts(texts)

        df = df.copy()
        df["sentiment_label"] = [r.label for r in results]
        df["sentiment_score"] = [r.score for r in results]
        df["sentiment_numeric"] = [r.numeric_score for r in results]
        return df


def aggregate_daily_sentiment(
    news_with_sentiment: pd.DataFrame,
    date_column: str = "date",
    sentiment_column: str = "sentiment_numeric",
) -> pd.DataFrame:
    """
    Aggregate sentiment scores per day (mean sentiment).
    """
    if date_column not in news_with_sentiment.columns:
        raise ValueError(f"Date column '{date_column}' not found")
    if sentiment_column not in news_with_sentiment.columns:
        raise ValueError(f"Sentiment column '{sentiment_column}' not found")

    df = news_with_sentiment.copy()
    df[date_column] = pd.to_datetime(df[date_column]).dt.date

    grouped = (
        df.groupby(date_column)[sentiment_column]
        .agg(["mean", "count"])
        .rename(columns={"mean": "daily_sentiment", "count": "num_headlines"})
        .reset_index()
    )
    return grouped


__all__ = ["FinBertSentimentAnalyzer", "aggregate_daily_sentiment", "SentimentResult"]

if __name__ == "__main__":

    print("Loading FinBERT model...")

    analyzer = FinBertSentimentAnalyzer()

    # Example financial headlines
    headlines = [
        "Apple stock surges after strong earnings report",
        "Tesla shares fall amid production concerns",
        "Microsoft announces new AI partnership",
        "Global markets remain stable despite inflation fears"
    ]

    print("\nAnalyzing sentiment...\n")

    results = analyzer.analyze_texts(headlines)

    for h, r in zip(headlines, results):
        print(f"Headline: {h}")
        print(f"Sentiment: {r.label} | Confidence: {r.score:.3f} | Numeric: {r.numeric_score:.3f}")
        print("-" * 60)
