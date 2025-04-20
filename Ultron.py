import pandas as pd
import yfinance as yf
import numpy as np
import datetime
import sys
import time
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

def get_available_dates(ticker, days=7):
    """Get a list of available trading dates for the past n days including today"""
    # Get current date
    today = datetime.datetime.now()
    today_str = today.strftime('%Y-%m-%d')
    print(f"Current date: {today_str}")
    
    # Get data for the past n+15 days (to account for weekends and holidays)
    end_date = today + datetime.timedelta(days=1)  # Add 1 day to include today
    start_date = end_date - datetime.timedelta(days=days+15)
    
    # Download daily data to get available trading days
    daily_data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    
    # Get all available dates in reverse chronological order (newest first)
    all_dates = daily_data.index.strftime('%Y-%m-%d').tolist()
    all_dates.reverse()  # Reverse to get newest first
    
    # Make sure we have today at the beginning if market is open
    if today_str not in all_dates:
        # Check if market might be open today (weekday)
        if today.weekday() < 5:  # 0-4 are Monday to Friday
            # Try to get intraday data for today
            today_data = yf.download(ticker, start=today_str, end=None, interval="1d", progress=False)
            if not today_data.empty:
                all_dates.insert(0, today_str)
    else:
        # Make sure today is first
        all_dates.remove(today_str)
        all_dates.insert(0, today_str)
    
    # Take only the requested number of days
    available_dates = all_dates[:days]
    
    print(f"Available dates (newest first): {available_dates}")
    return available_dates

def format_time_mst(dt):
    """Convert datetime to MST timezone and format in 12-hour format"""
    # Convert to MST timezone
    mst = pytz.timezone('US/Mountain')
    if dt.tzinfo is None:
        # Assume UTC for naive datetimes
        dt = pytz.utc.localize(dt)
    dt_mst = dt.astimezone(mst)
    # Format in 12-hour format
    return dt_mst.strftime('%I:%M %p MST')

class DashPlotter:
    def __init__(self, ticker="AAPL", days=7):
        self.ticker = ticker
        self.days = days
        self.available_dates = get_available_dates(ticker, days)
        self.selected_date = self.available_dates[0] if self.available_dates else datetime.datetime.now().strftime('%Y-%m-%d')
        self.trades = []
        self.all_trades = {}  # Dictionary to store trades for all dates
        self.df = None
        
    def analyze_data(self, date_str=None):
        """Analyze data for the given date"""
        if date_str is not None:
            self.selected_date = date_str
            
        print(f"Analyzing {self.ticker} for {self.selected_date}")
        
        # Download 5-minute interval data for the selected trading day
        start_date = pd.to_datetime(self.selected_date)
        
        # If analyzing today's data, use current time as end_date to get latest data
        today_str = datetime.datetime.now().strftime('%Y-%m-%d')
        if self.selected_date == today_str:
            # For today, always fetch the most recent data
            end_date = datetime.datetime.now() + datetime.timedelta(hours=1)  # Add buffer for latest data
            print(f"Fetching real-time data for today ({today_str})...")
            # Force a fresh download by setting a unique period
            self.df = yf.download(
                self.ticker, 
                start=start_date, 
                end=end_date, 
                interval="5m", 
                auto_adjust=False, 
                group_by='ticker',
                progress=False
            )
        else:
            # For historical dates, we can use a fixed end date
            end_date = start_date + datetime.timedelta(days=1)
            self.df = yf.download(
                self.ticker, 
                start=start_date, 
                end=end_date, 
                interval="5m", 
                auto_adjust=False, 
                group_by='ticker',
                progress=False
            )
        
        # Check if we got any data
        if self.df.empty:
            print(f"No data available for {self.ticker} on {self.selected_date}")
            return None
        
        # Fix column names
        self.df = self.df[self.ticker].copy() if self.ticker in self.df.columns else self.df
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
            # Get current candle data
            current_open = self.df['Open'].iloc[i]
            current_close = self.df['Close'].iloc[i]
            current_high = self.df['High'].iloc[i]
            current_low = self.df['Low'].iloc[i]
            
            # Check for the specific candle pattern conditions:
            # Condition 1: Red candle ((Close < Open) AND ((Close - Low) >= 2 * (Open - Close)) AND ((High - Open) <= 0.5 * (Close - Low)))
            # Condition 2: Green candle ((Close > Open) AND ((Open - Low) >= 2 * (Close - Open)) AND ((High - Close) <= 0.5 * (Open - Low)))
            red_candle_pattern = (current_close < current_open and 
                                 (current_close - current_low) >= 2 * (current_open - current_close) and 
                                 (current_high - current_open) <= 0.5 * (current_close - current_low))
            
            green_candle_pattern = (current_close > current_open and 
                                   (current_open - current_low) >= 2 * (current_close - current_open) and 
                                   (current_high - current_close) <= 0.5 * (current_open - current_low))
            
            # Buy if not in a position, the candle pattern is valid, price is above 6MA, and not selling on this candle
            if (not self.df['In_Position'].iloc[i] and 
                (red_candle_pattern) and 
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
        
        # Print trade summary
        print(f"\nTrading Summary for {self.ticker} on {self.selected_date}:")
        print(f"Number of trades: {len(self.trades)}")
        if self.trades:
            total_profit = sum(trade['profit'] for trade in self.trades)
            total_profit_pct = sum(trade['profit_pct'] for trade in self.trades)
            print(f"Total profit: ${total_profit:.2f} ({total_profit_pct:.2f}%)")
            print(f"Average profit per trade: ${total_profit/len(self.trades):.2f} ({total_profit_pct/len(self.trades):.2f}%)")
            
            # Count winning and losing trades
            winning_trades = sum(1 for trade in self.trades if trade['profit'] > 0)
            print(f"Win rate: {winning_trades/len(self.trades)*100:.2f}% ({winning_trades}/{len(self.trades)})")
        else:
            print("No trades were executed based on the strategy criteria.")
            print("This could be because no red candles closed above the 6MA.")
            
        return self.df
    
    def create_figure(self):
        """Create a plotly figure with the analyzed data"""
        if self.df is None or self.df.empty:
            # Return an empty figure with a message
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data available for {self.ticker} on {self.selected_date}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="#00FFFF")
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#000020",
                plot_bgcolor="#000020",
                font=dict(color="#00FFFF")
            )
            return fig
            
        # Create a subplot with 2 rows (price and volume)
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{self.ticker} - 5 Minute Candlestick Chart for {self.selected_date} (MST)', ''),
            row_heights=[0.7, 0.3]
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.df.index,
                open=self.df['Open'],
                high=self.df['High'],
                low=self.df['Low'],
                close=self.df['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add 6MA line
        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df['6MA'],
                line=dict(color='blue', width=2),
                name='6MA'
            ),
            row=1, col=1
        )
        
        # Add buy signals
        if self.df['Buy_Signal'].any():
            buy_indices = self.df[self.df['Buy_Signal']].index
            buy_prices = self.df.loc[buy_indices, 'Close']
            
            # Offset the buy signals downward by 0.3% of the price
            buy_offset = buy_prices * 0.003
            buy_y_positions = buy_prices - buy_offset
            
            fig.add_trace(
                go.Scatter(
                    x=buy_indices,
                    y=buy_y_positions,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name='Buy Signal'
                ),
                row=1, col=1
            )
        
        # Add sell signals
        if self.df['Sell_Signal'].any():
            sell_indices = self.df[self.df['Sell_Signal']].index
            sell_prices = self.df.loc[sell_indices, 'Close']
            
            # Offset the sell signals upward by 0.3% of the price
            sell_offset = sell_prices * 0.003
            sell_y_positions = sell_prices + sell_offset
            
            fig.add_trace(
                go.Scatter(
                    x=sell_indices,
                    y=sell_y_positions,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    name='Sell Signal'
                ),
                row=1, col=1
            )
        
        # Add volume bar chart
        fig.add_trace(
            go.Bar(
                x=self.df.index,
                y=self.df['Volume'],
                name='Volume',
                marker=dict(
                    color='rgba(0, 0, 255, 0.5)'
                )
            ),
            row=2, col=1
        )
        
        # Update layout with futuristic dark theme
        fig.update_layout(
            yaxis_title='Price ($)',
            height=800,
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#00FFFF")
            ),
            template="plotly_dark",
            paper_bgcolor="#000020",
            plot_bgcolor="#000020",
            font=dict(color="#00FFFF"),
            title_font=dict(color="#00FFFF", size=24),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update y-axis labels with futuristic styling
        fig.update_yaxes(
            title_text="Price ($)", 
            row=1, col=1, 
            gridcolor="#0F3460",
            zerolinecolor="#0F3460",
            tickfont=dict(color="#00FFFF")
        )
        fig.update_yaxes(
            title_text="Volume", 
            row=2, col=1, 
            gridcolor="#0F3460",
            zerolinecolor="#0F3460",
            tickfont=dict(color="#00FFFF")
        )
        # Get hourly ticks only
        mst = pytz.timezone('US/Mountain')
        hourly_ticks = []
        hourly_labels = []
        
        for dt in self.df.index:
            # Convert to MST
            if dt.tzinfo is None:
                dt_mst = pytz.utc.localize(dt).astimezone(mst)
            else:
                dt_mst = dt.astimezone(mst)
                
            # If this is on the hour (00 minutes), add it to hourly ticks
            if dt_mst.minute == 0:
                hourly_ticks.append(dt)
                hourly_labels.append(format_time_mst(dt).replace(' MST', ''))
        
        fig.update_xaxes(
            gridcolor="#0F3460",
            zerolinecolor="#0F3460",
            tickfont=dict(color="#00FFFF"),
            tickformat='%I:%M %p',  # 12-hour format
            hoverformat='%I:%M %p MST',  # 12-hour format with MST for hover
            # Use only hourly ticks
            tickvals=hourly_ticks,
            ticktext=hourly_labels
        )
        
        return fig
    
    def create_trades_table(self):
        """Create an HTML table with trade information"""
        if not self.trades:
            return html.Div("No trades executed on this date", 
                           style={'color': '#00FFFF', 'textAlign': 'center', 'padding': '20px', 'fontSize': '16px'})
        
        # Calculate total profit
        total_profit = sum(trade['profit'] for trade in self.trades)
        total_profit_pct = sum(trade['profit_pct'] for trade in self.trades)
        
        # Create table header
        header = html.Tr([
            html.Th("Time", style={'fontSize': '16px'}),
            html.Th("Price", style={'fontSize': '16px'}),
            html.Th("Profit ($)", style={'fontSize': '16px'}),
            html.Th("Profit (%)", style={'fontSize': '16px'})
        ])
        
        # Create table rows for each trade
        rows = []
        for trade in self.trades:
            row = html.Tr([
                html.Td(f"{format_time_mst(trade['entry_time'])} → {format_time_mst(trade['exit_time'])}", 
                       style={'fontSize': '18px'}),
                html.Td(f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f}", 
                       style={'fontSize': '16px'}),
                html.Td(f"${trade['profit']:.2f}", 
                       style={'color': 'green' if trade['profit'] > 0 else 'red', 'fontSize': '18px', 'fontWeight': 'bold'}),
                html.Td(f"{trade['profit_pct']:.2f}%", 
                       style={'color': 'green' if trade['profit_pct'] > 0 else 'red', 'fontSize': '18px', 'fontWeight': 'bold'})
            ])
            rows.append(row)
        
        # Add total row
        total_row = html.Tr([
            html.Td("TOTAL", style={'fontWeight': 'bold', 'fontSize': '16px'}),
            html.Td("", style={'fontSize': '16px'}),
            html.Td("", style={'fontSize': '16px'}),
            html.Td(f"{total_profit_pct:.2f}%", 
                   style={'fontWeight': 'bold', 'color': 'green' if total_profit_pct > 0 else 'red', 'fontSize': '20px'})
        ])
        
        # Create the table with futuristic styling
        table = html.Table(
            [header] + rows + [total_row],
            style={
                'width': '100%',
                'border-collapse': 'collapse',
                'margin-top': '20px',
                'margin-bottom': '20px',
                'color': '#00FFFF',
                'backgroundColor': '#000020',
                'borderRadius': '10px',
                'overflow': 'hidden',
                'boxShadow': '0 0 10px #00FFFF',
                'padding': '10px'
            }
        )
        
        return table
    
    def create_summary_stats(self):
        """Create a summary statistics component"""
        if not self.trades:
            return html.Div("No trades executed on this date", 
                           style={'color': '#00FFFF', 'textAlign': 'center', 'padding': '20px'})
        
        total_profit = sum(trade['profit'] for trade in self.trades)
        total_profit_pct = sum(trade['profit_pct'] for trade in self.trades)
        avg_profit = total_profit / len(self.trades)
        avg_profit_pct = total_profit_pct / len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade['profit'] > 0)
        win_rate = (winning_trades / len(self.trades)) * 100
        
        return html.Div([
            html.H4(f"Trading Summary for {self.ticker} on {self.selected_date}", 
                   style={'color': '#00FFFF', 'textAlign': 'center', 'marginBottom': '15px'}),
            html.Ul([
                html.Li(f"Number of trades: {len(self.trades)}", 
                       style={'color': '#00FFFF', 'marginBottom': '8px'}),
                html.Li([
                    "Total profit: ",
                    html.Span(f"{total_profit_pct:.2f}%", 
                              style={'color': '#00FF00' if total_profit > 0 else '#FF3333'})
                ], style={'marginBottom': '8px'}),
                html.Li(f"Win rate: {win_rate:.2f}% ({winning_trades}/{len(self.trades)})", 
                       style={'color': '#00FFFF'})
            ], style={'listStyleType': 'none', 'padding': '15px', 'backgroundColor': '#000030', 
                     'borderRadius': '10px', 'boxShadow': '0 0 5px #00FFFF'})
        ], style={'padding': '10px'})
    
    def create_multi_day_summary(self):
        """Create a summary statistics component for all analyzed dates"""
        # Analyze all dates if not already done
        if not self.all_trades:
            self.analyze_all_dates()
            
        # If still no trades, return a message
        if not self.all_trades:
            return html.Div("No trades found across all dates", 
                           style={'color': '#00FFFF', 'textAlign': 'center', 'padding': '20px'})
        
        # Flatten all trades into a single list
        all_trades_list = []
        for date_trades in self.all_trades.values():
            all_trades_list.extend(date_trades)
            
        # Calculate statistics
        total_trades = len(all_trades_list)
        total_profit_pct = sum(trade['profit_pct'] for trade in all_trades_list)
        winning_trades = sum(1 for trade in all_trades_list if trade['profit'] > 0)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Create a breakdown by date
        date_breakdown = []
        for date, trades in self.all_trades.items():
            date_profit_pct = sum(trade['profit_pct'] for trade in trades)
            date_win_rate = sum(1 for trade in trades if trade['profit'] > 0) / len(trades) * 100 if trades else 0
            
            date_breakdown.append(html.Li([
                f"{date}: ",
                html.Span(f"{date_profit_pct:.2f}%", 
                          style={'color': '#00FF00' if date_profit_pct > 0 else '#FF3333'}),
                f" ({len(trades)} trades, {date_win_rate:.1f}% win rate)"
            ], style={'color': '#00FFFF', 'marginBottom': '5px'}))
        
        return html.Div([
            html.H4(f"Multi-Day Trading Summary for {self.ticker}", 
                   style={'color': '#00FFFF', 'textAlign': 'center', 'marginBottom': '15px'}),
            html.Ul([
                html.Li(f"Total days analyzed: {len(self.all_trades)}", 
                       style={'color': '#00FFFF', 'marginBottom': '8px'}),
                html.Li(f"Total trades: {total_trades}", 
                       style={'color': '#00FFFF', 'marginBottom': '8px'}),
                html.Li([
                    "Total profit across all days: ",
                    html.Span(f"{total_profit_pct:.2f}%", 
                              style={'color': '#00FF00' if total_profit_pct > 0 else '#FF3333'})
                ], style={'marginBottom': '8px'}),
                html.Li(f"Overall win rate: {win_rate:.2f}% ({winning_trades}/{total_trades})", 
                       style={'color': '#00FFFF', 'marginBottom': '15px'})
            ], style={'listStyleType': 'none', 'padding': '15px', 'backgroundColor': '#000030', 
                     'borderRadius': '10px', 'boxShadow': '0 0 5px #00FFFF', 'marginBottom': '15px'}),
            
            html.H5("Daily Breakdown:", 
                   style={'color': '#00FFFF', 'marginBottom': '10px', 'marginLeft': '15px'}),
            html.Ul(date_breakdown, 
                   style={'listStyleType': 'none', 'padding': '15px', 'backgroundColor': '#000030', 
                          'borderRadius': '10px', 'boxShadow': '0 0 5px #00FFFF'})
        ], style={'padding': '10px'})
    
    def load_more_dates(self, days=None):
        """Load more historical dates"""
        if days is None:
            days = self.days * 2  # Double the number of days
        self.days = days
        self.available_dates = get_available_dates(self.ticker, days)
        return self.available_dates
        
    def analyze_all_dates(self):
        """Analyze data for all available dates and store the trades"""
        print(f"Analyzing all available dates for {self.ticker}...")
        
        # Store the current selected date to restore it later
        current_date = self.selected_date
        
        # Analyze each date and store the trades
        for date_str in self.available_dates:
            # Skip if we've already analyzed this date
            if date_str in self.all_trades:
                continue
                
            # Analyze the date
            self.analyze_data(date_str)
            
            # Store the trades for this date
            if self.trades:
                self.all_trades[date_str] = self.trades.copy()
        
        # Restore the originally selected date
        self.analyze_data(current_date)
        
        return self.all_trades

# Create the Dash app
def create_dash_app():
    # Create the Dash app with dark theme
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no"}
        ],
        suppress_callback_exceptions=True
    )
    
    # Define the ticker selection layout
    ticker_selection_layout = html.Div([
        html.Div([
            html.Div([
                html.H1("ULTRON",
                        style={
                            'textAlign': 'center',
                            'color': '#00FFFF',
                            'fontFamily': 'monospace',
                            'letterSpacing': '5px',
                            'textShadow': '0 0 20px #00FFFF',
                            'marginBottom': '40px',
                            'paddingTop': '60px',
                            'fontSize': '72px',
                            'width': '100%',
                            'display': 'flex',
                            'justifyContent': 'center'
                       }),
            ]),
                
            html.Div([
                html.Div([
                    html.Label("ENTER TICKER SYMBOL:", 
                              style={
                                  'color': '#00FFFF',
                                  'fontFamily': 'monospace',
                                  'fontSize': '22px',
                                  'textShadow': '0 0 8px #00FFFF',
                                  'marginBottom': '15px',
                                  'textAlign': 'center',
                                  'width': '100%'
                              }),
                    dcc.Input(
                        id='ticker-input',
                        type='text',
                        value='AAPL',
                        style={
                            'backgroundColor': '#000040',
                            'color': '#00FFFF',
                            'border': '3px solid #00FFFF',
                            'borderRadius': '8px',
                            'padding': '12px 20px',
                            'fontSize': '22px',
                            'width': '180px',
                            'textAlign': 'center',
                            'fontFamily': 'monospace',
                            'boxShadow': '0 0 15px #00FFFF',
                            'marginBottom': '25px'
                        }
                    ),
                ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'width': '100%'}),
                    
                html.Div([
                    html.Button('TRADE', id='analyze-button', n_clicks=0,
                               style={
                                   'backgroundColor': '#000080',
                                   'color': '#00FFFF',
                                   'border': '3px solid #00FFFF',
                                   'borderRadius': '8px',
                                   'padding': '12px 25px',
                                   'cursor': 'pointer',
                                   'boxShadow': '0 0 15px #00FFFF',
                                   'fontSize': '22px',
                                   'fontFamily': 'monospace',
                                   'fontWeight': 'bold',
                                   'marginBottom': '30px'
                               })
                ], style={'display': 'flex', 'justifyContent': 'center', 'width': '100%'})
            ], className='mobile-stack', style={
                'display': 'flex',
                'flexDirection': 'column',
                'alignItems': 'center',
                'justifyContent': 'center',
                'width': '100%'
            }),
            
            html.Div(id='loading-message', children=[
                html.Div("READY TO TRADE", 
                        style={
                            'color': '#00FF00',
                            'textAlign': 'center',
                            'fontFamily': 'monospace',
                            'fontSize': '20px',
                            'marginTop': '30px',
                            'animation': 'pulse 2s infinite',
                            'width': '100%',
                            'display': 'flex',
                            'justifyContent': 'center'
                        })
            ])
        ], style={
            'backgroundColor': '#000020',
            'padding': '60px',
            'borderRadius': '20px',
            'boxShadow': '0 0 40px rgba(0, 255, 255, 0.5)',
            'maxWidth': '1000px',
            'minHeight': '500px',
            'margin': '0 auto',
            'position': 'absolute',
            'top': '50%',
            'left': '50%',
            'transform': 'translate(-50%, -50%)',
            'width': '90%',
            'maxHeight': '90vh',
            'overflowY': 'auto'
        })
    ], style={
        'backgroundColor': '#000010',
        'minHeight': '100vh',
        'width': '100vw',
        'fontFamily': 'monospace',
        'position': 'relative',
        'display': 'flex',
        'justifyContent': 'center',
        'alignItems': 'center'
    })
    
    # Define the main app layout (initially hidden)
    main_app_layout = html.Div(id='main-app-container', style={'display': 'none'})
    
    # Combine layouts
    app.layout = html.Div([
        dcc.Store(id='ticker-store', data='AAPL'),
        # Auto-refresh interval (5 minutes = 300000 ms) - moved to top level
        dcc.Interval(
            id='auto-refresh-interval',
            interval=300000,  # in milliseconds
            n_intervals=0
        ),
        html.Div(id='ticker-selection-container', children=ticker_selection_layout),
        main_app_layout
    ])
    
    # Callback to handle ticker selection and initialize the main app
    @app.callback(
        [Output('ticker-selection-container', 'style'),
         Output('main-app-container', 'style'),
         Output('main-app-container', 'children'),
         Output('ticker-store', 'data'),
         Output('loading-message', 'children')],
        [Input('analyze-button', 'n_clicks')],
        [State('ticker-input', 'value')]
    )
    def initialize_app(n_clicks, ticker_value):
        if not n_clicks:
            # Initial state - show ticker selection, hide main app
            return (
                {'display': 'block'},  # Show ticker selection
                {'display': 'none'},   # Hide main app
                [],                    # Empty main app
                'SPXL',                # Default ticker
                [html.Div("READY TO TRADE", 
                         style={
                             'color': '#00FF00',
                             'textAlign': 'center',
                             'fontFamily': 'monospace',
                             'fontSize': '20px',
                             'marginTop': '20px',
                             'animation': 'pulse 2s infinite'
                         })]
            )
        
        # Show loading message
        loading_message = [
            html.Div("READY TO TRADE", 
                    style={
                        'color': '#00FF00',
                        'textAlign': 'center',
                        'fontFamily': 'monospace',
                        'fontSize': '20px',
                        'marginTop': '20px',
                        'animation': 'pulse 1s infinite'
                    })
        ]
        
        # Validate ticker
        ticker = ticker_value.strip().upper()
        if not ticker:
            ticker = 'SPXL'
            
        try:
            # Initialize the plotter with the selected ticker
            plotter = DashPlotter(ticker=ticker)
            
            # Analyze data for the default date
            plotter.analyze_data()
    
            # Create the main app layout with futuristic dark theme
            main_app = html.Div([
                # Hidden div for storing current date index
                html.Div(id='current-date-index', style={'display': 'none'}, children='0'),
                
                # Hidden div for storing available dates
                html.Div(id='available-dates-store', style={'display': 'none'}, 
                        children=','.join(plotter.available_dates)),
        
        
                # Header with futuristic styling
                html.Div([
                    html.Div([
                        html.Button('◀ BACK TO TICKER SELECTION', id='back-to-ticker-button', 
                                  style={
                                      'backgroundColor': '#000040',
                                      'color': '#00FFFF',
                                      'border': '1px solid #00FFFF',
                                      'borderRadius': '5px',
                                      'padding': '10px 15px',
                                      'marginRight': '15px',
                                      'cursor': 'pointer',
                                      'boxShadow': '0 0 5px #00FFFF',
                                      'fontSize': '14px'
                                  }),
                    ], style={'position': 'absolute', 'left': '20px', 'top': '20px'}),
                    
                    html.Div([
                        html.H1(f"{ticker} ULTRON", 
                               style={
                                    'textAlign': 'center',
                                    'color': '#00FFFF',
                                    'fontFamily': 'monospace',
                                    'letterSpacing': '3px',
                                    'textShadow': '0 0 15px #00FFFF',
                                    'marginBottom': '20px',
                                    'paddingTop': '70px',  # Further increased padding to move title down
                                    'fontSize': '42px'
                               }),
                    ]),
            
                    # Date navigation with back/next buttons and current date display
                    html.Div([
                        # First row: PREV, DATE, NEXT
                        html.Div([
                            html.Button('◀ PREV', id='prev-date-button', 
                                       style={
                                           'backgroundColor': '#000040',
                                           'color': '#00FFFF',
                                           'border': '1px solid #00FFFF',
                                           'borderRadius': '5px',
                                           'padding': '10px 15px',
                                           'marginRight': '15px',
                                           'cursor': 'pointer',
                                           'boxShadow': '0 0 5px #00FFFF'
                                       }),
                            html.Div(id='current-date-display', 
                                    children=f"DATE: {plotter.selected_date}",
                                    style={
                                        'color': '#00FFFF',
                                        'fontFamily': 'monospace',
                                        'fontSize': '14px',
                                        'padding': '8px 16px',
                                        'border': '1px solid #00FFFF',
                                        'borderRadius': '5px',
                                        'backgroundColor': '#000040',
                                        'boxShadow': '0 0 5px #00FFFF'
                                    }),
                            html.Button('NEXT ▶', id='next-date-button', 
                                       style={
                                           'backgroundColor': '#000020',  # Darker background for disabled
                                           'color': '#336666',  # Muted color for disabled
                                           'border': '1px solid #336666',
                                           'borderRadius': '5px',
                                           'padding': '10px 15px',
                                           'marginLeft': '15px',
                                           'cursor': 'not-allowed',  # Change cursor to indicate disabled
                                           'opacity': '0.5'  # Reduce opacity for disabled
                                       }),
                        ], style={
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'marginBottom': '15px',
                            'width': '100%'
                        }),
                        
                        # Second row: REFRESH and AUTO-REFRESH
                        html.Div([
                            html.Button('REFRESH', id='refresh-button', n_clicks=0, 
                                       style={
                                           'backgroundColor': '#000040',
                                           'color': '#00FFFF',
                                           'border': '1px solid #00FFFF',
                                           'borderRadius': '5px',
                                           'padding': '10px 15px',
                                           'marginRight': '15px',
                                           'cursor': 'pointer',
                                           'boxShadow': '0 0 5px #00FFFF'
                                       }),
                            html.Div(id='auto-refresh-indicator',
                                    children="AUTO-REFRESH: ACTIVE",
                                    style={
                                        'color': '#00FFFF',
                                        'fontFamily': 'monospace',
                                        'fontSize': '14px',
                                        'padding': '10px 15px',
                                        'border': '1px solid #00FFFF',
                                        'borderRadius': '5px',
                                        'backgroundColor': '#000040',
                                        'boxShadow': '0 0 5px #00FFFF'
                                    })
                        ], style={
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'width': '100%'
                        })
                    ], style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'alignItems': 'center',
                        'justifyContent': 'center',
                        'marginBottom': '20px',
                        'width': '100%'
                    }),
                ], style={
                    'backgroundColor': '#000020',
                    'padding': '10px',
                    'borderBottom': '2px solid #00FFFF',
                    'boxShadow': '0 5px 15px rgba(0, 255, 255, 0.3)',
                    'position': 'relative'
                }),
                
                # Main content area
                html.Div([
                    # Candlestick chart
                    dcc.Graph(
                        id='candlestick-chart', 
                        figure=plotter.create_figure(),
                        style={
                            'backgroundColor': '#000020',
                            'borderRadius': '10px',
                            'boxShadow': '0 0 10px rgba(0, 255, 255, 0.5)',
                            'marginBottom': '20px'
                        },
                        config={
                            'responsive': True,
                            'displayModeBar': True,
                            'scrollZoom': True
                        }
                    ),
                    
                    # Trade summary and statistics
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Div(id='multi-day-summary',
                                        children=plotter.create_multi_day_summary(),
                                        style={
                                            'width': '48%',
                                            'marginRight': '4%'
                                        }),
                                html.Div(id='trade-summary', 
                                        children=plotter.create_summary_stats(),
                                        style={
                                            'width': '48%'
                                        })
                            ], className='mobile-stack', style={
                                'display': 'flex',
                                'justifyContent': 'space-between',
                                'marginBottom': '20px'
                            })
                        ], className='mobile-stack', style={
                            'width': '48%',
                            'backgroundColor': '#000020',
                            'borderRadius': '10px',
                            'padding': '15px',
                            'boxShadow': '0 0 10px rgba(0, 255, 255, 0.5)'
                        }),
                        html.Div([
                            html.H4("Trades Executed", 
                                   style={
                                       'color': '#00FFFF', 
                                       'textAlign': 'center', 
                                       'marginBottom': '15px',
                                       'marginTop': '10px'  # Reduced from 20px to 15px
                                   }),
                            html.Div(id='trade-table', 
                                    children=plotter.create_trades_table(),
                                    style={'overflowX': 'auto'})
                        ], style={
                            'width': '48%',
                            'backgroundColor': '#000020',
                            'borderRadius': '10px',
                            'padding': '15px',
                            'boxShadow': '0 0 10px rgba(0, 255, 255, 0.5)'
                        })
                    ], className='mobile-stack', style={
                        'display': 'flex',
                        'justifyContent': 'space-between',
                        'marginBottom': '20px'
                    })
                ], style={
                    'padding': '20px',
                    'backgroundColor': '#000010'
                })
            ], style={
                'backgroundColor': '#000010',
                'minHeight': '100vh',
                'fontFamily': 'monospace'
            })
            
            # Return the updated UI components
            return (
                {'display': 'none'},     # Hide ticker selection
                {'display': 'block'},    # Show main app
                main_app,                # Main app content
                ticker,                  # Store ticker value
                loading_message          # Loading message
            )
        except Exception as e:
            # Log the error to console but don't show it in the UI
            print(f"ERROR: {str(e)}")
        
            # Show a generic message instead of the specific error
            loading_message = [
                html.Div("READY TO TRADE", 
                        style={
                            'color': '#00FF00',
                            'textAlign': 'center',
                            'fontFamily': 'monospace',
                            'fontSize': '20px',
                            'marginTop': '20px',
                            'animation': 'pulse 1s infinite'
                        })
            ]
        
            return (
                {'display': 'block'},    # Keep ticker selection visible
                {'display': 'none'},     # Hide main app
                [],                      # Empty main app
                ticker,                  # Store ticker value
                loading_message          # Loading message (generic, not error)
            )
    
    # Callback to go back to ticker selection
    @app.callback(
        [Output('ticker-selection-container', 'style', allow_duplicate=True),
         Output('main-app-container', 'style', allow_duplicate=True)],
        [Input('back-to-ticker-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def back_to_ticker_selection(n_clicks):
        if n_clicks:
            return {'display': 'block'}, {'display': 'none'}
        raise PreventUpdate
    
    # Callback to update the current date display and next button style
    @app.callback(
        [Output('current-date-display', 'children'),
         Output('next-date-button', 'style')],
        [Input('current-date-index', 'children'),
         Input('available-dates-store', 'children'),
         Input('ticker-store', 'data')]
    )
    def update_date_display(current_index, available_dates_str, ticker):
        available_dates = available_dates_str.split(',')
        current_index = int(current_index)
        
        if current_index < 0 or current_index >= len(available_dates):
            current_index = len(available_dates) - 1
            
        selected_date = available_dates[current_index]
        
        # Determine if we're on the most recent date (index 0)
        # If so, disable the next button by graying it out
        if current_index == 0:
            next_button_style = {
                'backgroundColor': '#000020',  # Darker background for disabled
                'color': '#336666',  # Muted color for disabled
                'border': '1px solid #336666',
                'borderRadius': '5px',
                'padding': '10px 15px',
                'marginLeft': '15px',
                'cursor': 'not-allowed',  # Change cursor to indicate disabled
                'opacity': '0.5'  # Reduce opacity for disabled
            }
        else:
            next_button_style = {
                'backgroundColor': '#000040',
                'color': '#00FFFF',
                'border': '1px solid #00FFFF',
                'borderRadius': '5px',
                'padding': '10px 15px',
                'marginLeft': '15px',
                'cursor': 'pointer',
                'boxShadow': '0 0 5px #00FFFF'
            }
            
        return f"DATE: {selected_date}", next_button_style
    
    # Callback for date navigation buttons
    @app.callback(
        [Output('current-date-index', 'children'),
         Output('available-dates-store', 'children')],
        [Input('prev-date-button', 'n_clicks'),
         Input('next-date-button', 'n_clicks'),
         Input('refresh-button', 'n_clicks'),
         Input('auto-refresh-interval', 'n_intervals')],
        [State('current-date-index', 'children'),
         State('available-dates-store', 'children'),
         State('ticker-store', 'data')]
    )
    def navigate_dates(prev_clicks, next_clicks, refresh_clicks, n_intervals, current_index, available_dates_str, ticker):
        ctx = dash.callback_context
        
        if not ctx.triggered:
            raise PreventUpdate
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        available_dates = available_dates_str.split(',')
        current_index = int(current_index)
        
        if trigger_id == 'prev-date-button' and prev_clicks:
            # Move to previous date (next in the list since our list is newest first)
            if current_index < len(available_dates) - 1:
                current_index += 1
        elif trigger_id == 'next-date-button' and next_clicks:
            # Move to next date (previous in the list since our list is newest first)
            if current_index > 0:
                current_index -= 1
        elif trigger_id in ['refresh-button', 'auto-refresh-interval']:
            # Refresh available dates with newest first
            available_dates = get_available_dates(ticker, 7)
            
            # If we're viewing today (index 0), stay there after refresh
            if current_index == 0:
                current_index = 0
            # Otherwise, try to find the date we were viewing in the new list
            else:
                old_date = available_dates_str.split(',')[current_index]
                if old_date in available_dates:
                    current_index = available_dates.index(old_date)
                else:
                    # If date not found, reset to most recent
                    current_index = 0
                
        return str(current_index), ','.join(available_dates)
    
    # Define callback for chart updates
    @app.callback(
        [Output('candlestick-chart', 'figure'),
         Output('trade-summary', 'children'),
         Output('trade-table', 'children'),
         Output('multi-day-summary', 'children')],
        [Input('current-date-index', 'children'),
         Input('refresh-button', 'n_clicks'),
         Input('auto-refresh-interval', 'n_intervals'),
         Input('next-date-button', 'n_clicks')],  # Add next button to trigger refresh
        [State('available-dates-store', 'children'),
         State('ticker-store', 'data')]
    )
    def update_chart(current_index, n_clicks, n_intervals, next_clicks, available_dates_str, ticker):
        available_dates = available_dates_str.split(',')
        current_index = int(current_index)
        
        if current_index < 0 or current_index >= len(available_dates):
            current_index = 0
            
        selected_date = available_dates[current_index]
        
        # Create a plotter for the current ticker and analyze data
        plotter = DashPlotter(ticker=ticker)
        
        # Analyze the current date
        plotter.analyze_data(selected_date)
        
        # Create the figure and trade information for current date
        fig = plotter.create_figure()
        summary = plotter.create_summary_stats()
        table = plotter.create_trades_table()
        
        # Analyze all dates in the background and create multi-day summary
        # This will populate plotter.all_trades
        multi_day_summary = plotter.create_multi_day_summary()
        
        return fig, summary, table, multi_day_summary
    
    # Callback to update the auto-refresh indicator
    @app.callback(
        Output('auto-refresh-indicator', 'style'),
        [Input('auto-refresh-interval', 'n_intervals')]
    )
    def update_refresh_indicator(n_intervals):
        # Flash the indicator when refreshing
        if n_intervals % 2 == 0:
            return {
                'color': '#00FFFF',
                'fontFamily': 'monospace',
                'fontSize': '14px',
                'padding': '10px 15px',
                'marginLeft': '15px',
                'border': '1px solid #00FFFF',
                'borderRadius': '5px',
                'backgroundColor': '#000080',  # Highlight color
                'boxShadow': '0 0 10px #00FFFF',
                'transition': 'all 0.5s ease'
            }
        else:
            return {
                'color': '#00FFFF',
                'fontFamily': 'monospace',
                'fontSize': '14px',
                'padding': '10px 15px',
                'marginLeft': '15px',
                'border': '1px solid #00FFFF',
                'borderRadius': '5px',
                'backgroundColor': '#000040',
                'boxShadow': '0 0 5px #00FFFF',
                'transition': 'all 0.5s ease'
            }
    
    return app

# Run the app
#if __name__ == "__main__":
    #app = create_dash_app()
    #print("Starting Dash server for candlestick analysis...")
    #print("Open your web browser and navigate to http://127.0.0.1:8050/")
    #app.run_server(host="0.0.0.0", port=8080)app.run(debug=True, use_reloader=False)
