import pandas as pd
from datetime import datetime
import random

tickers = ['AAPL', 'TSLA', 'AMZN']
sources = ['Reuters', 'Bloomberg', 'CNBC', 'WSJ', 'Financial Times']

start_date = datetime(2025, 1, 1)
end_date = datetime(2026, 3, 24)
dates = pd.date_range(start_date, end_date)

headlines_templates = {
    'AAPL': [
        'Apple shares {action} as {event}',
        'iPhone {product} {impact} sales',
        'Apple faces {issue} in {area}',
        'Analysts {opinion} Apple {focus}',
        '{product_line} drives Apple growth'
    ],
    'TSLA': [
        'Tesla {action} after {event}',
        'Tesla Autopilot {update} raises questions',
        'EV market {trend} for Tesla',
        'Tesla {achievement} in battery tech',
        'Elon Musk announces Tesla {plan}'
    ],
    'AMZN': [
        'Amazon {action} {service} expansion',
        'AWS {growth} amid cloud demand',
        'Amazon e-commerce {change} competition',
        'Prime membership {impact} revenue',
        'Amazon invests in {sector}'
    ]
}

actions = ['rise', 'fall', 'surge', 'dip', 'climb', 'drop']
events = ['earnings beat', 'supply chain issues', 'product launch', 'analyst upgrade']
products = ['16 launch', '17 Pro', 'new chip']
impacts = ['boosts', 'hurts', 'supports']
issues = ['regulatory scrutiny', 'competition pressure']
areas = ['supply chain', 'China market', 'services']
opinions = ['raise targets for', 'downgrade']
focuses = ['vision pro', 'stock', 'earnings']
product_lines = ['Services', 'iPad', 'Mac']
updates = ['progress', 'scrutiny']
trends = ['growth favors', 'slowdown hits']
achievements = ['breakthrough', 'milestone']
plans = ['expansion', 'robotaxi']
services_list = ['Prime', 'delivery', 'logistics']
growths = ['accelerates', 'slows']
changes = ['faces', 'leads']
sectors = ['AI', 'logistics', 'healthcare']

def generate_headline(ticker):
    temp = random.choice(headlines_templates[ticker])
    placeholders = {
        'action': random.choice(actions),
        'event': random.choice(events),
        'product': random.choice(products),
        'impact': random.choice(impacts),
        'issue': random.choice(issues),
        'area': random.choice(areas),
        'opinion': random.choice(opinions),
        'focus': random.choice(focuses),
        'product_line': random.choice(product_lines),
        'update': random.choice(updates),
        'trend': random.choice(trends),
        'achievement': random.choice(achievements),
        'plan': random.choice(plans),
        'service': random.choice(services_list),
        'growth': random.choice(growths),
        'change': random.choice(changes),
        'sector': random.choice(sectors)
    }
    try:
        return temp.format(**{k: v for k, v in placeholders.items() if '{' + k + '}' in temp})
    except KeyError:
        return temp.format(**placeholders)  # fallback

rows = []
for date in dates:
    for ticker in tickers:
        for _ in range(2):
            headline = generate_headline(ticker)
            source = random.choice(sources)
            rows.append({
                'date': date.strftime('%Y-%m-%d'),
                'ticker': ticker,
                'headline': headline,
                'source': source
            })

df = pd.DataFrame(rows)
df.to_csv('temp_sample_news.csv', index=False)
print(f'Generated temp_sample_news.csv with {len(df)} rows.')
print('First 12 rows:')
print(df.head(12).to_csv(index=False))
