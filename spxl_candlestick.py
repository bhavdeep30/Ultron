import mplfinance as mpf
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.widgets import RadioButtons, Button
import datetime
import sys

TICKER = "SPXL"

def get_available_dates(days=7):
    """Get a list of available trading dates for the past n days"""
    # Get data for the past n+5 days (to account for weekends and holidays)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days+5)
    
    # Download daily data to get available trading days
    daily_data = yf.download(TICKER, start=start_date, end=end_date, interval="1d")
    
    # Get the last n trading days
    available_dates = daily_data.index[-days:].strftime('%Y-%m-%d').tolist()
    return available_dates

class InteractivePlotter:
    def __init__(self, days=7):
        self.days = days
        self.available_dates = get_available_dates(days)
        self.selected_date = self.available_dates[-1]  # Default to latest date
        self.fig = None
        self.axes = None
        self.table_ax = None
        self.main_ax = None
        self.trades = []
        self.df = None
        
    def load_more_dates(self, days=None):
        """Load more historical dates"""
        if days is None:
            days = self.days * 2  # Double the number of days
        self.days = days
        self.available_dates = get_available_dates(days)
        self.update_plot()
        
    def update_plot(self, event=None):
        """Update the plot with the currently selected date"""
        if self.fig is not None:
            plt.close(self.fig)
        
        print(f"Analyzing {TICKER} for {self.selected_date}")
        
        # Download 5-minute interval data for the selected trading day
        # Use a 2-day period to ensure we get the full trading day
        start_date = pd.to_datetime(self.selected_date)
        end_date = start_date + datetime.timedelta(days=1)
        self.df = yf.download(TICKER, start=start_date, end=end_date, interval="5m", auto_adjust=False, group_by='ticker')
        
        # Check if we got any data
        if self.df.empty:
            print(f"No data available for {TICKER} on {self.selected_date}")
            # Create a simple figure with a message
            self.fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, f"No data available for {TICKER} on {self.selected_date}", 
                    horizontalalignment='center', verticalalignment='center', fontsize=14)
            ax.axis('off')
            
            # Add date selection radio buttons
            radio_ax = self.fig.add_axes([0.01, 0.5, 0.1, 0.3])
            self.radio = RadioButtons(radio_ax, self.available_dates)
            active_idx = self.available_dates.index(self.selected_date) if self.selected_date in self.available_dates else -1
            if active_idx >= 0:
                self.radio.set_active(active_idx)
            self.radio.on_clicked(self.select_date)
            
            # Add refresh button
            refresh_ax = self.fig.add_axes([0.01, 0.4, 0.1, 0.05])
            self.refresh_button = Button(refresh_ax, 'Refresh')
            self.refresh_button.on_clicked(self.update_plot)
            
            plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.25)
            plt.draw()
            return
        
        # Fix column names
        self.df = self.df[TICKER].copy()
        self.df.dropna(inplace=True)
        
        # Ensure numeric types
        self.df = self.df.astype({
            "Open": "float",
            "High": "float",
            "Low": "float",
            "Close": "float",
            "Volume": "float"
        })
        
        # Set index name
        self.df.index.name = 'Date'
        
        # Calculate the 6-period moving average
        self.df['6MA'] = self.df['Close'].rolling(window=6).mean()
        
        # Identify red candles (close < open)
        self.df['Red_Candle'] = self.df['Close'] < self.df['Open']
        
        # Initialize buy/sell signals
        self.df['Buy_Signal'] = False
        self.df['Sell_Signal'] = False
        self.df['In_Position'] = False
        
        # Initialize trade tracking
        self.trades = []
        current_position = None
        
        # Implement the trading strategy
        for i in range(6, len(self.df)):  # Start after 6MA is available
            # If not in a position and we have a red candle that closes above 6MA
            # Also check that we're not selling on this candle (which would happen if previous candle had a buy)
            if (not self.df['In_Position'].iloc[i] and 
                self.df['Red_Candle'].iloc[i] and 
                self.df['Close'].iloc[i] > self.df['6MA'].iloc[i] and
                not self.df['Sell_Signal'].iloc[i]):
                
                # Buy signal
                self.df.loc[self.df.index[i], 'Buy_Signal'] = True
                self.df.loc[self.df.index[i], 'In_Position'] = True
                
                # Record the buy
                entry_price = self.df['Close'].iloc[i]
                entry_time = self.df.index[i]
                current_position = {'entry_time': entry_time, 'entry_price': entry_price}
                
                # If this is not the last candle, sell at the next candle
                if i + 1 < len(self.df):
                    # Sell at the next candle
                    self.df.loc[self.df.index[i+1], 'Sell_Signal'] = True
                    
                    # Record the sell and calculate profit
                    exit_price = self.df['Close'].iloc[i+1]
                    exit_time = self.df.index[i+1]
                    
                    profit = exit_price - entry_price
                    profit_pct = (profit / entry_price) * 100
                    
                    self.trades.append({
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
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate total return
        total_profit = 0
        total_profit_pct = 0
        if self.trades:
            total_profit = sum(trade['profit'] for trade in self.trades)
            total_profit_pct = sum(trade['profit_pct'] for trade in self.trades)
        
        # Create addplots with the 6MA
        ap = [mpf.make_addplot(self.df['6MA'], color='blue', width=1.2)]
        
        # Only add buy signals if there are any
        if self.df['Buy_Signal'].any():
            buy_signals = np.where(self.df['Buy_Signal'], self.df['Close'], np.nan)
            ap.append(mpf.make_addplot(buy_signals, type='scatter', marker='^', markersize=100, color='g'))
        
        # Only add sell signals if there are any
        if self.df['Sell_Signal'].any():
            sell_signals = np.where(self.df['Sell_Signal'], self.df['Close'], np.nan)
            ap.append(mpf.make_addplot(sell_signals, type='scatter', marker='v', markersize=100, color='r'))
        
        # Create the figure and primary axis
        self.fig, self.axes = mpf.plot(
            self.df,
            type='candle',
            volume=True,
            style='yahoo',
            title=f'{TICKER} - 5 Minute Candlestick Chart for {self.selected_date}',
            ylabel='Price ($)',
            ylabel_lower='Volume',
            figratio=(12, 8),
            figscale=1.2,
            addplot=ap,
            returnfig=True,
            tight_layout=False  # Disable tight_layout to avoid warning
        )
        
        self.main_ax = self.axes[0]
        
        # Add a table below the chart
        if self.trades:
            # Format the trades table data
            table_data = []
            for i, trade in enumerate(self.trades):
                table_data.append([
                    f"{trade['entry_time'].strftime('%H:%M')} → {trade['exit_time'].strftime('%H:%M')}",
                    f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f}",
                    f"${trade['profit']:.2f}",
                    f"{trade['profit_pct']:.2f}%"
                ])
            
            # Add a row for total profit
            table_data.append(["TOTAL", "", f"${total_profit:.2f}", f"{total_profit_pct:.2f}%"])
            
            # Create a new axis for the table
            self.fig.set_size_inches(12, 10)  # Make figure taller to accommodate table
            self.table_ax = self.fig.add_axes([0.1, 0.05, 0.8, 0.2])  # [left, bottom, width, height]
            self.table_ax.axis('off')
            
            # Create the table
            table = Table(self.table_ax, bbox=[0, 0, 1, 1])
            
            # Add column headers
            headers = ['Time', 'Price', 'Profit ($)', 'Profit (%)']
            for i, header in enumerate(headers):
                table.add_cell(0, i, 0.2, 0.1, text=header, loc='center', facecolor='lightgrey')
            
            # Add data rows
            for i, row in enumerate(table_data):
                row_color = 'white' if i < len(self.trades) else 'lightgrey'
                for j, cell in enumerate(row):
                    table.add_cell(i+1, j, 0.2, 0.1, text=cell, loc='center', facecolor=row_color)
            
            self.table_ax.add_table(table)
        
        # Print trade summary
        print(f"\nTrading Summary for {TICKER} on {self.selected_date}:")
        print(f"Number of trades: {len(self.trades)}")
        if self.trades:
            print(f"Total profit: ${total_profit:.2f} ({total_profit_pct:.2f}%)")
            print(f"Average profit per trade: ${total_profit/len(self.trades):.2f} ({total_profit_pct/len(self.trades):.2f}%)")
            
            # Count winning and losing trades
            winning_trades = sum(1 for trade in self.trades if trade['profit'] > 0)
            print(f"Win rate: {winning_trades/len(self.trades)*100:.2f}% ({winning_trades}/{len(self.trades)})")
        else:
            print("No trades were executed based on the strategy criteria.")
            print("This could be because no red candles closed above the 6MA.")
        
        # Add date selection radio buttons
        radio_ax = self.fig.add_axes([0.01, 0.5, 0.1, 0.3])
        self.radio = RadioButtons(radio_ax, self.available_dates)
        # Set the active radio button to match the current selected date
        active_idx = self.available_dates.index(self.selected_date) if self.selected_date in self.available_dates else -1
        if active_idx >= 0:
            self.radio.set_active(active_idx)
        self.radio.on_clicked(self.select_date)
        
        # Add refresh button
        refresh_ax = self.fig.add_axes([0.01, 0.4, 0.1, 0.05])
        self.refresh_button = Button(refresh_ax, 'Refresh')
        self.refresh_button.on_clicked(self.update_plot)
        
        # Add load more dates button
        more_dates_ax = self.fig.add_axes([0.01, 0.35, 0.1, 0.05])
        self.more_dates_button = Button(more_dates_ax, 'More Dates')
        self.more_dates_button.on_clicked(lambda event: self.load_more_dates())
        
        # Use figure-level adjustments instead of tight_layout
        plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.25)
        plt.draw()
    
    def select_date(self, date_str):
        """Handle date selection from radio buttons"""
        self.selected_date = date_str
        print(f"Selected date: {date_str}")
        self.update_plot()
    
    def show(self):
        """Display the interactive plot"""
        self.update_plot()
        plt.show()

# Create and show the interactive plotter
if __name__ == "__main__":
    plotter = InteractivePlotter()
    plotter.show()

# This section has been moved into the InteractivePlotter class
