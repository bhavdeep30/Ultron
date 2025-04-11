import mplfinance as mpf
import pandas as pd
import yfinance as yf

TICKER = "SPXL"

# Download 5-minute interval data for latest trading day
df = yf.download(TICKER, period="1d", interval="5m", auto_adjust=False, group_by='ticker')

# Fix column names
df = df[TICKER].copy()
df.dropna(inplace=True)

# Ensure numeric types
df = df.astype({
    "Open": "float",
    "High": "float",
    "Low": "float",
    "Close": "float",
    "Volume": "float"
})

# Set index name
df.index.name = 'Date'

# Calculate the 6-period moving average
df['6MA'] = df['Close'].rolling(window=6).mean()

# Create addplot for the 6MA
ap = mpf.make_addplot(df['6MA'], color='blue', width=1.2)

# Plot with labels and the 6MA
mpf.plot(
    df,
    type='candle',
    volume=True,
    style='yahoo',
    title=f'{TICKER} - 5 Minute Candlestick Chart (1D)',
    ylabel='Price ($)',
    ylabel_lower='Volume',
    figratio=(12, 6),
    figscale=1.2,
    addplot=ap
)
