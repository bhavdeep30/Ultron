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
    def __init__(self, ticker="SPXL", days=7):
        self.ticker = ticker
        self.days = days
        self.available_dates = get_available_dates(ticker, days)
        self.selected_date = self.available_dates[0] if self.available_dates else datetime.datetime.now().strftime('%Y-%m-%d')
        self.trades = []
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
            
            fig.add_trace(
                go.Scatter(
                    x=buy_indices,
                    y=buy_prices,
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
            
            fig.add_trace(
                go.Scatter(
                    x=sell_indices,
                    y=sell_prices,
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
            title=f'{self.ticker} - 5 Minute Candlestick Chart for {self.selected_date} (MST)',
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
                           style={'color': '#00FFFF', 'textAlign': 'center', 'padding': '20px'})
        
        # Calculate total profit
        total_profit = sum(trade['profit'] for trade in self.trades)
        total_profit_pct = sum(trade['profit_pct'] for trade in self.trades)
        
        # Create table header
        header = html.Tr([
            html.Th("Time"),
            html.Th("Price"),
            html.Th("Profit ($)"),
            html.Th("Profit (%)")
        ])
        
        # Create table rows for each trade
        rows = []
        for trade in self.trades:
            row = html.Tr([
                html.Td(f"{format_time_mst(trade['entry_time'])} → {format_time_mst(trade['exit_time'])}"),
                html.Td(f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f}"),
                html.Td(f"${trade['profit']:.2f}", style={'color': 'green' if trade['profit'] > 0 else 'red'}),
                html.Td(f"{trade['profit_pct']:.2f}%", style={'color': 'green' if trade['profit_pct'] > 0 else 'red'})
            ])
            rows.append(row)
        
        # Add total row
        total_row = html.Tr([
            html.Td("TOTAL", style={'font-weight': 'bold'}),
            html.Td(""),
            html.Td(f"${total_profit:.2f}", style={'font-weight': 'bold', 'color': 'green' if total_profit > 0 else 'red'}),
            html.Td(f"{total_profit_pct:.2f}%", style={'font-weight': 'bold', 'color': 'green' if total_profit_pct > 0 else 'red'})
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
                'boxShadow': '0 0 10px #00FFFF'
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
                    html.Span(f"${total_profit:.2f} ({total_profit_pct:.2f}%)", 
                              style={'color': '#00FF00' if total_profit > 0 else '#FF3333'})
                ], style={'marginBottom': '8px'}),
                html.Li(f"Win rate: {win_rate:.2f}% ({winning_trades}/{len(self.trades)})", 
                       style={'color': '#00FFFF'})
            ], style={'listStyleType': 'none', 'padding': '15px', 'backgroundColor': '#000030', 
                     'borderRadius': '10px', 'boxShadow': '0 0 5px #00FFFF'})
        ], style={'padding': '10px'})
    
    def load_more_dates(self, days=None):
        """Load more historical dates"""
        if days is None:
            days = self.days * 2  # Double the number of days
        self.days = days
        self.available_dates = get_available_dates(self.ticker, days)
        return self.available_dates

# Create the Dash app
def create_dash_app():
    # Create the Dash app with dark theme
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ]
    )
    
    # Define the ticker selection layout
    ticker_selection_layout = html.Div([
        html.Div([
            html.H1("ULTRON MARKET TRADE", 
                   style={
                       'textAlign': 'center',
                       'color': '#00FFFF',
                       'fontFamily': 'monospace',
                       'letterSpacing': '3px',
                       'textShadow': '0 0 10px #00FFFF',
                       'marginBottom': '40px',
                       'paddingTop': '40px'
                   }),
            
            html.Div([
                html.Label("ENTER TICKER SYMBOL:", 
                          style={
                              'color': '#00FFFF',
                              'fontFamily': 'monospace',
                              'fontSize': '18px',
                              'marginRight': '15px',
                              'textShadow': '0 0 5px #00FFFF'
                          }),
                dcc.Input(
                    id='ticker-input',
                    type='text',
                    value='SPXL',
                    style={
                        'backgroundColor': '#000040',
                        'color': '#00FFFF',
                        'border': '2px solid #00FFFF',
                        'borderRadius': '5px',
                        'padding': '10px 15px',
                        'fontSize': '18px',
                        'width': '150px',
                        'textAlign': 'center',
                        'fontFamily': 'monospace',
                        'boxShadow': '0 0 10px #00FFFF'
                    }
                ),
                html.Button('ANALYZE', id='analyze-button', n_clicks=0,
                           style={
                               'backgroundColor': '#000080',
                               'color': '#00FFFF',
                               'border': '2px solid #00FFFF',
                               'borderRadius': '5px',
                               'padding': '10px 20px',
                               'marginLeft': '15px',
                               'cursor': 'pointer',
                               'boxShadow': '0 0 10px #00FFFF',
                               'fontSize': '18px',
                               'fontFamily': 'monospace',
                               'fontWeight': 'bold'
                           })
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'marginBottom': '40px'
            }),
            
            html.Div(id='loading-message', children=[
                html.Div("READY TO ANALYZE", 
                        style={
                            'color': '#00FFFF',
                            'textAlign': 'center',
                            'fontFamily': 'monospace',
                            'fontSize': '16px',
                            'marginTop': '20px',
                            'animation': 'pulse 2s infinite'
                        })
            ])
        ], style={
            'backgroundColor': '#000020',
            'padding': '40px',
            'borderRadius': '15px',
            'boxShadow': '0 0 30px rgba(0, 255, 255, 0.3)',
            'maxWidth': '800px',
            'margin': '100px auto'
        })
    ], style={
        'backgroundColor': '#000010',
        'minHeight': '100vh',
        'fontFamily': 'monospace'
    })
    
    # Define the main app layout (initially hidden)
    main_app_layout = html.Div(id='main-app-container', style={'display': 'none'})
    
    # Combine layouts
    app.layout = html.Div([
        dcc.Store(id='ticker-store', data='SPXL'),
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
                [html.Div("READY TO ANALYZE", 
                         style={
                             'color': '#00FFFF',
                             'textAlign': 'center',
                             'fontFamily': 'monospace',
                             'fontSize': '16px',
                             'marginTop': '20px',
                             'animation': 'pulse 2s infinite'
                         })]
            )
        
        # Show loading message
        loading_message = [
            html.Div("INITIALIZING DATA ANALYSIS...", 
                    style={
                        'color': '#00FF00',
                        'textAlign': 'center',
                        'fontFamily': 'monospace',
                        'fontSize': '16px',
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
        
                # Auto-refresh interval (5 minutes = 300000 ms)
                dcc.Interval(
                    id='auto-refresh-interval',
                    interval=300000,  # in milliseconds
                    n_intervals=0
                ),
        
                # Header with futuristic styling
                html.Div([
                    html.Div([
                        html.Button('◀ BACK TO TICKER SELECTION', id='back-to-ticker-button', 
                                  style={
                                      'backgroundColor': '#000040',
                                      'color': '#00FFFF',
                                      'border': '1px solid #00FFFF',
                                      'borderRadius': '5px',
                                      'padding': '8px 15px',
                                      'marginRight': '15px',
                                      'cursor': 'pointer',
                                      'boxShadow': '0 0 5px #00FFFF',
                                      'fontSize': '12px'
                                  }),
                    ], style={'position': 'absolute', 'left': '20px', 'top': '20px'}),
                    
                    html.H1(f"{ticker} ULTRON TRADE", 
                           style={
                               'textAlign': 'center',
                               'color': '#00FFFF',
                               'fontFamily': 'monospace',
                               'letterSpacing': '3px',
                               'textShadow': '0 0 10px #00FFFF',
                               'marginBottom': '20px',
                               'paddingTop': '20px'
                           }),
            
                    # Date navigation with back/next buttons and current date display
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
                                    'fontSize': '18px',
                                    'padding': '10px 20px',
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
                        html.Button('REFRESH', id='refresh-button', n_clicks=0, 
                                   style={
                                       'backgroundColor': '#000040',
                                       'color': '#00FFFF',
                                       'border': '1px solid #00FFFF',
                                       'borderRadius': '5px',
                                       'padding': '10px 15px',
                                       'marginLeft': '15px',
                                       'cursor': 'pointer',
                                       'boxShadow': '0 0 5px #00FFFF'
                                   }),
                        html.Div(id='auto-refresh-indicator',
                                children="AUTO-REFRESH: ACTIVE",
                                style={
                                    'color': '#00FFFF',
                                    'fontFamily': 'monospace',
                                    'fontSize': '14px',
                                    'padding': '5px 10px',
                                    'marginLeft': '15px',
                                    'border': '1px solid #00FFFF',
                                    'borderRadius': '5px',
                                    'backgroundColor': '#000040',
                                    'boxShadow': '0 0 5px #00FFFF'
                                })
                    ], style={
                        'display': 'flex', 
                        'alignItems': 'center', 
                        'justifyContent': 'center',
                        'marginBottom': '20px'
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
                        }
                    ),
                    
                    # Trade summary and statistics
                    html.Div([
                        html.Div(id='trade-summary', 
                                children=plotter.create_summary_stats(),
                                style={
                                    'width': '48%',
                                    'backgroundColor': '#000020',
                                    'borderRadius': '10px',
                                    'padding': '15px',
                                    'boxShadow': '0 0 10px rgba(0, 255, 255, 0.5)'
                                }),
                        html.Div(id='trade-table', 
                                children=plotter.create_trades_table(),
                                style={
                                    'width': '48%',
                                    'backgroundColor': '#000020',
                                    'borderRadius': '10px',
                                    'padding': '15px',
                                    'boxShadow': '0 0 10px rgba(0, 255, 255, 0.5)'
                                })
                    ], style={
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
            # If there's an error, show error message and keep ticker selection visible
            error_message = [
                html.Div(f"ERROR: {str(e)}", 
                        style={
                            'color': '#FF3333',
                            'textAlign': 'center',
                            'fontFamily': 'monospace',
                            'fontSize': '16px',
                            'marginTop': '20px'
                        })
            ]
            return (
                {'display': 'block'},    # Keep ticker selection visible
                {'display': 'none'},     # Hide main app
                [],                      # Empty main app
                ticker,                  # Store ticker value
                error_message            # Error message
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
         Output('trade-table', 'children')],
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
        plotter.analyze_data(selected_date)
        
        # Create the figure and trade information
        fig = plotter.create_figure()
        summary = plotter.create_summary_stats()
        table = plotter.create_trades_table()
        
        return fig, summary, table
    
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
                'padding': '5px 10px',
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
                'padding': '5px 10px',
                'marginLeft': '15px',
                'border': '1px solid #00FFFF',
                'borderRadius': '5px',
                'backgroundColor': '#000040',
                'boxShadow': '0 0 5px #00FFFF',
                'transition': 'all 0.5s ease'
            }
    
    return app

# Run the app
if __name__ == "__main__":
    app = create_dash_app()
    print("Starting Dash server for candlestick analysis...")
    print("Open your web browser and navigate to http://127.0.0.1:8050/")
    app.run(debug=True, use_reloader=False)
