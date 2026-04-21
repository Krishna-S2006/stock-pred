# Knowledge Base

This document explains how the project works end to end: raw stock data and news are collected, news is scored with FinBERT, technical features are engineered, the merged dataset is turned into supervised sequences, and an LSTM predicts the next-day price direction.

## Project Overview

The architecture is intentionally simple and linear:

1. `data_collection.py` downloads stock prices and fetches news.
2. `sentiment_model.py` scores headlines with FinBERT and aggregates them by day.
3. `feature_engineering.py` builds price features and combines them with sentiment.
4. `lstm_model.py` converts the tabular dataset into sequences and trains/predicts with an LSTM.
5. `models/train_model.py` runs the full training pipeline and saves model weights.
6. `dashboard/app.py` runs the same pipeline inside a Streamlit UI.

At a high level:

```text
Stock prices + News headlines
            |
            v
      FinBERT sentiment
            |
            v
   Daily sentiment aggregation
            |
            v
   Feature engineering / merge
            |
            v
   Supervised dataset creation
            |
            v
     Sequence creation for LSTM
            |
            v
   PriceMovementLSTM training
            |
            v
   Next-day up/down prediction
```

## Sentiment Model

The sentiment layer lives in `src/sentiment_model.py` and is built around FinBERT, a finance-tuned BERT model from Hugging Face.

### Main Classes and Functions

`SentimentResult` is a small dataclass that stores the raw model label, the confidence score, and a numeric score used downstream.

```python
@dataclass
class SentimentResult:
    label: str
    score: float
    numeric_score: float
```

`FinBertSentimentAnalyzer` wraps the transformers pipeline.

```python
class FinBertSentimentAnalyzer:
    def __init__(self, model_name: str = FINBERT_MODEL_NAME, device: int = -1) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
```

The analyzer converts labels into a signed numeric sentiment score:

```python
@staticmethod
def _to_numeric(label: str, score: float) -> float:
    label = label.upper()
    if "POSITIVE" in label:
        return float(score)
    if "NEGATIVE" in label:
        return float(-score)
    return 0.0
```

That mapping is the key bridge between NLP output and the numeric feature pipeline:

- Positive headline -> positive float score
- Negative headline -> negative float score
- Neutral headline -> 0.0

### Text Scoring Flow

`analyze_texts()` accepts any iterable of strings, sends them through the transformer pipeline, and returns a list of `SentimentResult` objects.

```python
def analyze_texts(self, texts: Iterable[str]) -> List[SentimentResult]:
    outputs = self._pipeline(list(texts), truncation=True)
    results: List[SentimentResult] = []
    for out in outputs:
        label = out["label"]
        score = float(out["score"])
        numeric = self._to_numeric(label, score)
        results.append(SentimentResult(label=label, score=score, numeric_score=numeric))
    return results
```

`score_dataframe()` adds three columns to a news DataFrame:

- `sentiment_label`
- `sentiment_score`
- `sentiment_numeric`

```python
def score_dataframe(self, df: pd.DataFrame, text_column: str = "headline") -> pd.DataFrame:
    texts = df[text_column].astype(str).tolist()
    results = self.analyze_texts(texts)

    df = df.copy()
    df["sentiment_label"] = [r.label for r in results]
    df["sentiment_score"] = [r.score for r in results]
    df["sentiment_numeric"] = [r.numeric_score for r in results]
    return df
```

### Daily Aggregation

`aggregate_daily_sentiment()` compresses article-level sentiment into one row per date using the mean sentiment and headline count.

```python
def aggregate_daily_sentiment(
    news_with_sentiment: pd.DataFrame,
    date_column: str = "date",
    sentiment_column: str = "sentiment_numeric",
) -> pd.DataFrame:
    df = news_with_sentiment.copy()
    df[date_column] = pd.to_datetime(df[date_column]).dt.date

    grouped = (
        df.groupby(date_column)[sentiment_column]
        .agg(["mean", "count"])
        .rename(columns={"mean": "daily_sentiment", "count": "num_headlines"})
        .reset_index()
    )
    return grouped
```

### Example Usage

```python
from src.sentiment_model import FinBertSentimentAnalyzer, aggregate_daily_sentiment

analyzer = FinBertSentimentAnalyzer()
news_scored = analyzer.score_dataframe(news_df, text_column="headline")
daily_sentiment = aggregate_daily_sentiment(news_scored)
```

Expected output shape:

- Input: one row per headline
- Output: one row per date with `daily_sentiment` and `num_headlines`

### Practical Notes

- The model is loaded from `ProsusAI/finbert`.
- `device=-1` means CPU by default.
- The code uses truncation, so long headlines or text inputs are clipped safely.

## LSTM Model

The sequence model is implemented in `src/lstm_model.py`.

### Sequence Creation

`create_sequences()` converts a 2D feature matrix and target vector into overlapping sliding windows.

```python
def create_sequences(features: np.ndarray, targets: np.ndarray, seq_len: int = 5):
    xs = []
    ys = []

    if len(features) <= seq_len:
        return torch.empty(0), torch.empty(0)

    for i in range(len(features) - seq_len):
        xs.append(features[i : i + seq_len])
        ys.append(targets[i + seq_len])

    if len(xs) == 0:
        return torch.empty(0), torch.empty(0)

    x_arr = np.stack(xs)
    y_arr = np.array(ys)

    return torch.from_numpy(x_arr).float(), torch.from_numpy(y_arr).long()
```

Conceptually, if `seq_len=5`, each training sample contains the previous 5 days of features and the label is the movement on the next day.

### Model Architecture

`PriceMovementLSTM` is a compact binary classifier.

```python
class PriceMovementLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        logits = self.fc(last_hidden)
        return logits
```

Architecture summary:

- Input shape: `(batch, seq_len, input_size)`
- LSTM produces hidden states for each timestep
- The last timestep is passed into a fully connected layer
- Output logits represent two classes: `down` and `up`

### Training Container

`LSTMTrainingResult` packages the trained model together with the fitted scaler and metadata needed for inference.

```python
@dataclass
class LSTMTrainingResult:
    model: PriceMovementLSTM
    scaler: StandardScaler
    feature_columns: Sequence[str]
    seq_len: int
```

This matters because prediction needs the exact same feature columns, sequence length, and normalization parameters used in training.

### Training Flow

`train_lstm_on_dataframe()` is the main training entry point.

```python
def train_lstm_on_dataframe(
    df_supervised: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str = "target_up",
    seq_len: int = 5,
    test_size: float = 0.2,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> LSTMTrainingResult:
```

Step-by-step behavior:

1. Drop rows with missing feature or target values.
2. Extract the numeric feature matrix and labels.
3. Fit a `StandardScaler` on all features.
4. Split the scaled data into train and test sets without shuffling.
5. Turn both splits into sequences with `create_sequences()`.
6. Fail fast if training sequences cannot be formed.
7. Build `PriceMovementLSTM`, `CrossEntropyLoss`, and Adam optimizer.
8. Train for the requested number of epochs.
9. Optionally evaluate on the test sequence set.
10. Return the model, scaler, feature names, and sequence length.

Important implementation detail: the split uses `shuffle=False`, which preserves time order and avoids leaking future data into the past.

### Prediction Flow

`predict_next_movement()` is the inference helper.

```python
def predict_next_movement(
    training_result: LSTMTrainingResult,
    recent_features: pd.DataFrame,
) -> Tuple[int, float]:
```

It works like this:

1. Select the feature columns stored in `training_result`.
2. Drop rows with missing values.
3. Require at least `seq_len` rows.
4. Take the last `seq_len` rows as the input window.
5. Apply the fitted scaler.
6. Run the model in evaluation mode.
7. Softmax the logits into probabilities.
8. Return the predicted label and the probability of an upward move.

```python
pred_label, prob_up = predict_next_movement(training_result, supervised)
```

### Example Usage

```python
from src.lstm_model import train_lstm_on_dataframe, predict_next_movement

training_result = train_lstm_on_dataframe(supervised, feature_columns=["daily_return", "ma_close", "Volume", "daily_sentiment"])
label, prob_up = predict_next_movement(training_result, supervised)
```

## Feature Engineering

The feature pipeline is in `src/feature_engineering.py` and has three responsibilities: normalize the price data, merge sentiment, and create a supervised target.

### Price Feature Computation

`compute_price_features()` prepares raw OHLCV data returned by Yahoo Finance.

```python
def compute_price_features(price_df: pd.DataFrame, ma_window: int = 5) -> pd.DataFrame:
```

Pipeline breakdown:

1. Copy the input frame.
2. Reset the index if the date is stored in a `DatetimeIndex`.
3. Flatten MultiIndex columns if needed.
4. Normalize column names to lowercase.
5. Rename `date`, `close`, and `volume` into the expected internal schema.
6. Validate that `date`, `Close`, and `Volume` exist.
7. Convert date and numeric columns to clean types.
8. Sort by date.
9. Compute `daily_return` using percentage change.
10. Compute `ma_close` using a rolling mean window.

Example snippet:

```python
df["daily_return"] = df["Close"].pct_change()
df["ma_close"] = df["Close"].rolling(window=ma_window, min_periods=1).mean()
```

### Merging Sentiment

`merge_price_and_sentiment()` joins the stock and sentiment tables on `date`.

```python
def merge_price_and_sentiment(
    price_features: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
) -> pd.DataFrame:
```

Behavior:

- Validates that sentiment contains the expected columns.
- Normalizes both date columns to Python `date` objects.
- Left-joins sentiment onto price features.
- Fills missing sentiment with `0.0`.
- Fills missing headline counts with `0`.

This keeps days without news in the dataset instead of dropping them.

### Supervised Dataset Creation

`build_supervised_dataset()` creates the target label used by the LSTM.

```python
def build_supervised_dataset(
    merged_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
```

Logic:

- Sort by date.
- Verify `Close` exists.
- Verify every requested feature column exists.
- Create `next_close` by shifting `Close` by one day forward.
- Create `target_up` as `1` when the next close is higher than today, otherwise `0`.
- Drop the last row because it has no next-day target.

```python
df["next_close"] = df["Close"].shift(-1)
df["target_up"] = (df["next_close"] > df["Close"]).astype(int)
```

### Example Usage

```python
price_features = compute_price_features(prices, ma_window=5)
merged = merge_price_and_sentiment(price_features, daily_sentiment)
supervised = build_supervised_dataset(merged, feature_columns=["daily_return", "ma_close", "Volume", "daily_sentiment"])
```

## Training Pipeline

`models/train_model.py` runs the full offline training flow. It is the clearest reference for how all modules fit together outside the UI.

### Step-by-Step Flow

1. Parse command-line arguments for `ticker`, `start`, and `end`.
2. Call `run_pipeline(ticker, start, end)`.
3. Download historical prices with `download_stock_data()`.
4. Load sample news from `load_sample_news()` and filter it with `filter_news()`.
5. If no news is available, build a neutral daily sentiment frame.
6. Otherwise, score headlines with `FinBertSentimentAnalyzer`.
7. Aggregate the scored headlines with `aggregate_daily_sentiment()`.
8. Compute technical features with `compute_price_features()`.
9. Merge sentiment with price features.
10. Build the supervised dataset and define feature columns.
11. Train the LSTM with `train_lstm_on_dataframe()`.
12. Predict the next-day direction with `predict_next_movement()`.
13. Save the model state dict to `models/lstm_<TICKER>.pt`.

### Key Code Path

```python
prices = download_stock_data(ticker, start, end)
news = load_sample_news()
news_filtered = filter_news(news, ticker=ticker, start=start, end=end)

if news_filtered.empty:
    daily_sentiment = pd.DataFrame(
        {"date": pd.to_datetime(prices["date"]).dt.date, "daily_sentiment": 0.0, "num_headlines": 0}
    ).drop_duplicates(subset=["date"])
else:
    analyzer = FinBertSentimentAnalyzer()
    news_scored = analyzer.score_dataframe(news_filtered, text_column="headline")
    daily_sentiment = aggregate_daily_sentiment(news_scored)

price_features = compute_price_features(prices, ma_window=5)
merged = merge_price_and_sentiment(price_features, daily_sentiment)
supervised = build_supervised_dataset(merged, feature_columns=feature_cols)
training_result = train_lstm_on_dataframe(supervised, feature_columns=feature_cols, target_column="target_up", seq_len=5, epochs=10, lr=1e-3, batch_size=16)
pred_label, prob_up = predict_next_movement(training_result, supervised)
```

### Why This File Matters

This script is the best reference for training because it shows the intended production order of operations, not just isolated utilities.

## Dashboard Integration

`dashboard/app.py` reuses the same core modules, but presents them through Streamlit and Plotly.

### UI Flow

1. Set up the Streamlit page.
2. Let the user choose `ticker`, `start`, and `end`.
3. Download and normalize price data.
4. Load live news with `fetch_real_news()`.
5. Score news with FinBERT if headlines exist.
6. Aggregate sentiment by date.
7. Visualize price and sentiment.
8. Build the supervised dataset.
9. Train the LSTM directly in the dashboard session.
10. Display the next-day prediction as a metric.

### Important Integration Details

The dashboard contains a helper called `normalize_price_dataframe()` that handles different Yahoo Finance shapes before feature engineering.

```python
def normalize_price_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})

    return df
```

The dashboard uses the live news path:

- `fetch_real_news(ticker)` from `src/data_collection.py`
- `FinBertSentimentAnalyzer` from `src/sentiment_model.py`
- `aggregate_daily_sentiment()` for the chart and model input

If no news is found, it falls back to neutral sentiment for each date so the pipeline can still run.

### Dashboard Prediction Flow

```python
price_features = compute_price_features(prices, ma_window=5)
merged = merge_price_and_sentiment(price_features, daily_sentiment)
supervised = build_supervised_dataset(merged, feature_columns=feature_cols)
training_result = train_lstm_on_dataframe(supervised, feature_columns=feature_cols, seq_len=5, epochs=5)
pred_label, prob_up = predict_next_movement(training_result, supervised)
```

The result is shown with `st.metric()` as `UP` or `DOWN` with the probability of an upward move.

## Data Flow Diagram

| Stage | Input | Output | File |
|---|---|---|---|
| Stock download | Ticker, date range | OHLCV table | `src/data_collection.py` |
| News fetch | Ticker | News headlines | `src/data_collection.py` |
| Sentiment scoring | Headlines | Per-headline sentiment | `src/sentiment_model.py` |
| Sentiment aggregation | Scored headlines | Daily sentiment table | `src/sentiment_model.py` |
| Price features | OHLCV table | Technical features | `src/feature_engineering.py` |
| Merge | Price features + daily sentiment | Combined daily dataset | `src/feature_engineering.py` |
| Supervised labeling | Combined daily dataset | `target_up` dataset | `src/feature_engineering.py` |
| Sequence creation | Feature rows | LSTM windows | `src/lstm_model.py` |
| Training | Sequences + labels | Trained model + scaler | `src/lstm_model.py`, `models/train_model.py` |
| Prediction | Latest feature window | Up/down forecast | `src/lstm_model.py` |

## Key Files Summary

| File | Purpose |
|---|---|
| `src/data_collection.py` | Downloads stock data, determines default date ranges, and fetches live news from Yahoo Finance. |
| `src/preprocessing.py` | Loads sample news CSV data and filters it by ticker and date range. |
| `src/sentiment_model.py` | Loads FinBERT, scores headlines, and aggregates sentiment by day. |
| `src/feature_engineering.py` | Builds technical indicators, merges sentiment, and creates the supervised target. |
| `src/lstm_model.py` | Defines the LSTM classifier, training loop, sequence creation, and prediction helper. |
| `models/train_model.py` | Runs the end-to-end offline training pipeline and saves model weights. |
| `dashboard/app.py` | Streamlit dashboard that visualizes data, sentiment, and predictions. |
| `generate_stock_data.py` | Utility script for producing stock data used in the project. |
| `generate_dummy_news.py` | Utility script for producing sample or synthetic news data. |
| `notebooks/analysis.ipynb` | Exploratory notebook for analysis and pipeline experimentation. |
| `data/sample_news.csv` | Sample headline dataset used for local training and demos. |
| `data/stock_data.csv` | Raw stock history used in feature engineering demos. |
| `data/final_dataset.csv` | Saved merged supervised dataset ready for modeling. |
| `models/lstm_AAPL.pt` | Saved model weights for the AAPL example. |
| `README.md` | Short project summary and quick-start instructions. |
| `TODO.md` | Notes about recent implementation work and dashboard fixes. |

## Practical Example: End-to-End Flow

The following is the canonical sequence the project uses:

```python
prices = download_stock_data("AAPL", "2023-01-01", "2023-03-31")
news = fetch_real_news("AAPL")

analyzer = FinBertSentimentAnalyzer()
news_scored = analyzer.score_dataframe(news, text_column="headline")
daily_sentiment = aggregate_daily_sentiment(news_scored)

price_features = compute_price_features(prices, ma_window=5)
merged = merge_price_and_sentiment(price_features, daily_sentiment)
supervised = build_supervised_dataset(merged, feature_columns=["daily_return", "ma_close", "Volume", "daily_sentiment"])

training_result = train_lstm_on_dataframe(supervised, feature_columns=["daily_return", "ma_close", "Volume", "daily_sentiment"])
pred_label, prob_up = predict_next_movement(training_result, supervised)
```

## Implementation Notes

- The project is designed around daily granularity, not intraday modeling.
- Sentiment is aggregated by day, so the model sees one sentiment signal per trading day.
- The target is binary: next day up or down.
- The scaler is part of the inference contract, so it must be preserved alongside the model.
- The dashboard retrains on each run; this is fine for demos but not ideal for a production app.
