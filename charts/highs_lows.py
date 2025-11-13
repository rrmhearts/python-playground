import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import alpaca_trade_api as tradeapi

# === Alpaca API credentials ===
API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"
BASE_URL = "https://data.alpaca.markets/v2"  # Data endpoint

# === Connect to Alpaca ===
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# === Fetch data ===
symbol = "AAPL"
timeframe = "1Day"
limit = 500  # number of data points

bars = api.get_bars(symbol, timeframe, limit=limit).df
bars = bars[bars['symbol'] == symbol]  # Filter if needed
bars.index = pd.to_datetime(bars.index)

# === Find local highs and lows ===
window = 10  # how far to look for relative extrema
bars['local_max'] = bars['high'][(bars['high'] == bars['high'].iloc[argrelextrema(bars['high'].values, np.greater_equal, order=window)[0]])]
bars['local_min'] = bars['low'][(bars['low'] == bars['low'].iloc[argrelextrema(bars['low'].values, np.less_equal, order=window)[0]])]

# Drop NaNs for line fitting
max_points = bars.dropna(subset=['local_max'])
min_points = bars.dropna(subset=['local_min'])

# === Fit lines through highs and lows ===
# We'll fit a simple linear regression (in log scale)
x = np.arange(len(bars))

# Highs
if len(max_points) > 1:
    coeffs_high = np.polyfit(x[max_points.index.get_indexer(bars.index)], np.log(max_points['local_max']), 1)
    trend_high = np.exp(np.polyval(coeffs_high, x))
else:
    trend_high = np.nan

# Lows
if len(min_points) > 1:
    coeffs_low = np.polyfit(x[min_points.index.get_indexer(bars.index)], np.log(min_points['local_min']), 1)
    trend_low = np.exp(np.polyval(coeffs_low, x))
else:
    trend_low = np.nan

# === Plot ===
plt.figure(figsize=(12, 6))
plt.plot(bars.index, bars['close'], label="Close", color='blue')
plt.scatter(max_points.index, max_points['local_max'], color='red', label='Local Highs')
plt.scatter(min_points.index, min_points['local_min'], color='green', label='Local Lows')

# Draw fitted lines
if isinstance(trend_high, np.ndarray):
    plt.plot(bars.index, trend_high, '--', color='red', label='High Trendline')
if isinstance(trend_low, np.ndarray):
    plt.plot(bars.index, trend_low, '--', color='green', label='Low Trendline')

plt.yscale('log')
plt.title(f"{symbol} Local Highs/Lows Trendlines (Log Scale)")
plt.xlabel("Date")
plt.ylabel("Price (log scale)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
