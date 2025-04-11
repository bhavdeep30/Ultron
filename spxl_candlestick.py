import mplfinance as mpf
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

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

# Identify red candles (close < open)
df['Red_Candle'] = df['Close'] < df['Open']

# Initialize buy/sell signals
df['Buy_Signal'] = False
df['Sell_Signal'] = False
df['In_Position'] = False

# Initialize trade tracking
trades = []
current_position = None

# Implement the trading strategy
for i in range(6, len(df)):  # Start after 6MA is available
    # If not in a position and we have a red candle that closes above 6MA
    if (not df['In_Position'].iloc[i] and 
        df['Red_Candle'].iloc[i] and 
        df['Close'].iloc[i] > df['6MA'].iloc[i]):
        
        # Buy signal
        df.loc[df.index[i], 'Buy_Signal'] = True
        df.loc[df.index[i], 'In_Position'] = True
        
        # Record the buy
        entry_price = df['Close'].iloc[i]
        entry_time = df.index[i]
        current_position = {'entry_time': entry_time, 'entry_price': entry_price}
        
        # If this is not the last candle, sell at the next candle
        if i + 1 < len(df):
            # Sell at the next candle
            df.loc[df.index[i+1], 'Sell_Signal'] = True
            
            # Record the sell and calculate profit
            exit_price = df['Close'].iloc[i+1]
            exit_time = df.index[i+1]
            
            profit = exit_price - entry_price
            profit_pct = (profit / entry_price) * 100
            
            trades.append({
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': exit_time,
                'exit_price': exit_price,
                'profit': profit,
                'profit_pct': profit_pct
            })
            
            # Reset position after selling
            current_position = None
        else:
            # If this is the last candle, we can't sell
            print("Warning: Buy signal on the last candle - no sell possible")

# Create a DataFrame for trades
trades_df = pd.DataFrame(trades)

# Calculate total return
total_profit = 0
if trades:
    total_profit = sum(trade['profit'] for trade in trades)
    total_profit_pct = sum(trade['profit_pct'] for trade in trades)

# Create addplots with the 6MA
ap = [mpf.make_addplot(df['6MA'], color='blue', width=1.2)]

# Only add buy signals if there are any
if df['Buy_Signal'].any():
    buy_signals = np.where(df['Buy_Signal'], df['Close'], np.nan)
    ap.append(mpf.make_addplot(buy_signals, type='scatter', marker='^', markersize=100, color='g'))

# Only add sell signals if there are any
if df['Sell_Signal'].any():
    sell_signals = np.where(df['Sell_Signal'], df['Close'], np.nan)
    ap.append(mpf.make_addplot(sell_signals, type='scatter', marker='v', markersize=100, color='r'))

# Create the figure and primary axis
fig, axes = mpf.plot(
    df,
    type='candle',
    volume=True,
    style='yahoo',
    title=f'{TICKER} - 5 Minute Candlestick Chart with Trading Strategy',
    ylabel='Price ($)',
    ylabel_lower='Volume',
    figratio=(12, 8),
    figscale=1.2,
    addplot=ap,
    returnfig=True
)

# Add a table below the chart
if trades:
    # Format the trades table data
    table_data = []
    for i, trade in enumerate(trades):
        table_data.append([
            f"{trade['entry_time'].strftime('%H:%M')} → {trade['exit_time'].strftime('%H:%M')}",
            f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f}",
            f"${trade['profit']:.2f}",
            f"{trade['profit_pct']:.2f}%"
        ])
    
    # Add a row for total profit
    table_data.append(["TOTAL", "", f"${total_profit:.2f}", f"{total_profit_pct:.2f}%"])
    
    # Create a new axis for the table
    fig.set_size_inches(12, 10)  # Make figure taller to accommodate table
    table_ax = fig.add_axes([0.1, 0.05, 0.8, 0.2])  # [left, bottom, width, height]
    table_ax.axis('off')
    
    # Create the table
    table = Table(table_ax, bbox=[0, 0, 1, 1])
    
    # Add column headers
    headers = ['Time', 'Price', 'Profit ($)', 'Profit (%)']
    for i, header in enumerate(headers):
        table.add_cell(0, i, 0.2, 0.1, text=header, loc='center', facecolor='lightgrey')
    
    # Add data rows
    for i, row in enumerate(table_data):
        row_color = 'white' if i < len(trades) else 'lightgrey'
        for j, cell in enumerate(row):
            table.add_cell(i+1, j, 0.2, 0.1, text=cell, loc='center', facecolor=row_color)
    
    table_ax.add_table(table)

# Print trade summary
print(f"\nTrading Summary for {TICKER}:")
print(f"Number of trades: {len(trades)}")
if trades:
    print(f"Total profit: ${total_profit:.2f} ({total_profit_pct:.2f}%)")
    print(f"Average profit per trade: ${total_profit/len(trades):.2f} ({total_profit_pct/len(trades):.2f}%)")
    
    # Count winning and losing trades
    winning_trades = sum(1 for trade in trades if trade['profit'] > 0)
    print(f"Win rate: {winning_trades/len(trades)*100:.2f}% ({winning_trades}/{len(trades)})")
else:
    print("No trades were executed based on the strategy criteria.")
    print("This could be because no red candles closed above the 6MA.")

plt.tight_layout()
plt.show()
