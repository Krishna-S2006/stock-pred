import pandas as pd
import numpy as np
from datetime import datetime
import random

# Date range matching news data
start_date = datetime(2025, 1, 1)
end_date = datetime(2026, 3, 24)
dates = pd.date_range(start_date, end_date)
tickers = ['AAPL', 'TSLA', 'AMZN']

def generate_stock_row(date, ticker):
    """Generate realistic OHLCV data based on historical patterns"""
    # Base prices with ticker-specific trends
    base_price = {'AAPL': 200, 'TSLA': 250, 'AMZN': 180}[ticker]
    
    # Daily volatility and trend
    trend = random.uniform(-0.02, 0.02)
    volatility = random.uniform(0.01, 0.04)
    
    # Generate OHLCV
    open_price = base_price * np.exp(np.cumsum(np.random.normal(trend, volatility, 1))[0])
    high_price = open_price * random.uniform(1.005, 1.05)
    low_price = open_price * random.uniform(0.95, 0.995)
    close_price = open_price + random.uniform(low_price - open_price, high_price - open_price)
    adj_close = close_price * random.uniform(0.98, 1.02)  # slight adjustment
    volume = random.randint(30_000_000, 150_000_000)
    
    return {
        'date': date.strftime('%Y-%m-%d'),
        'Adj Close': round(adj_close, 6),
        'Close': round(close_price, 2),
        'High': round(high_price, 2),
        'Low': round(low_price, 2),
        'Open': round(open_price, 2),
        'Volume': volume
    }

rows = []
for date in dates:
    for ticker in tickers:
        row = generate_stock_row(date, ticker)
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv('temp_stock_data.csv', index=False)
print(f'Generated temp_stock_data.csv with {len(df)} rows for {dates.shape[0]} days x 3 tickers.')
print('Sample:')
print(df.head(12).to_csv(index=False))
