# Financial News Stock Prediction

An end-to-end ML pipeline that predicts stock price movements using financial news sentiment analysis (FinBERT), technical indicators, and LSTM neural networks. Includes an interactive Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-brightgreen)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org/)

## 🏗️ Project Structure

```
financial-news-stock-prediction/
├── data/
│   ├── final_dataset.csv       # Processed features + labels
│   ├── sample_news.csv         # Sample financial news headlines
│   └── stock_data.csv          # Raw stock OHLCV data
├── src/
│   ├── data_collection.py      # Yahoo Finance data downloader
│   ├── preprocessing.py        # News text processing
│   ├── sentiment_model.py      # FinBERT sentiment analyzer
│   ├── feature_engineering.py  # Technical indicators + sentiment aggregation
│   └── lstm_model.py           # LSTM model implementation
├── dashboard/
│   └── app.py                  # Streamlit + Plotly dashboard
├── models/
│   ├── lstm_AAPL.pt            # Pre-trained LSTM model (AAPL)
│   ├── train_model.py          # Training script
│   └── tempCodeRunnerFile.py   # (Temporary/ignore)
├── notebooks/
│   └── analysis.ipynb          # EDA and pipeline demo
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Launch dashboard**:
```bash
streamlit run dashboard/app.py
```
Select ticker (AAPL, TSLA, AMZN) and date range to see predictions!

## 📊 Features

- **Sentiment Analysis**: FinBERT transformer on financial news headlines
- **Technical Features**: Returns, moving averages, volume ratios
- **LSTM Prediction**: Next-day price direction (up/down)
- **Interactive Dashboard**: Stock charts, sentiment trends, predictions
- **Production-ready**: Pre-trained model + processed dataset included

## 🧪 Usage Examples

**Train new model**:
```bash
python models/train_model.py --ticker AAPL --start 2022-01-01 --end 2023-12-31
```

**Load pre-trained model + predict**:
```python
from src.lstm_model import load_model, predict_next_day
model = load_model('models/lstm_AAPL.pt')
prediction = predict_next_day(model, features_df)
```

## 📈 Example Output

**Dashboard Screenshot** (add your screenshot here):
![Dashboard](screenshots/dashboard.png)

**Sample Prediction**:
```
AAPL (2023-12-29) → Next day: UP (85% confidence)
```

## 🔧 Development

- Edit `src/` modules to extend features
- Add real-time news APIs to `data_collection.py`
- Train on more tickers: modify `train_model.py`
- Deploy dashboard: `streamlit deploy`

## 📦 Datasets

- `sample_news.csv`: 1000+ headlines with dates/tickers
- `final_dataset.csv`: Ready-to-use features + labels
- `stock_data.csv`: Historical OHLCV for multiple tickers

## Issues & Troubleshooting

- **Model download slow**: First run downloads FinBERT (~500MB)
- **CUDA errors**: Install CPU version or set `device='cpu'`
- **No news data**: Uses sample dataset; extend for live news

## Next Steps

- [ ] Integrate live news APIs (Alpha Vantage, NewsAPI)
- [ ] Multi-ticker ensemble model
- [ ] Backtesting framework
- [ ] Model interpretability (SHAP)

---
**Built with ❤️ for financial ML enthusiasts**

