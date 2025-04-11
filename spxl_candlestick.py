import pandas as pd
import yfinance as yf
import numpy as np
import datetime
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

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

class DashPlotter:
    def __init__(self, days=7):
        self.days = days
        self.available_dates = get_available_dates(days)
        self.selected_date = self.available_dates[-1]  # Default to latest date
        self.trades = []
        self.df = None
        
    def analyze_data(self, date_str=None):
        """Analyze data for the given date"""
        if date_str is not None:
            self.selected_date = date_str
            
        print(f"Analyzing {TICKER} for {self.selected_date}")
        
        # Download 5-minute interval data for the selected trading day
        # Use a 2-day period to ensure we get the full trading day
        start_date = pd.to_datetime(self.selected_date)
        end_date = start_date + datetime.timedelta(days=1)
        self.df = yf.download(TICKER, start=start_date, end=end_date, interval="5m", auto_adjust=False, group_by='ticker')
        
        # Check if we got any data
        if self.df.empty:
            print(f"No data available for {TICKER} on {self.selected_date}")
            return None
        
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
        
        # Print trade summary
        print(f"\nTrading Summary for {TICKER} on {self.selected_date}:")
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
                text=f"No data available for {TICKER} on {self.selected_date}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig
            
        # Create a subplot with 2 rows (price and volume)
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{TICKER} - 5 Minute Candlestick Chart for {self.selected_date}', ''),
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
        
        # Update layout
        fig.update_layout(
            title=f'{TICKER} - 5 Minute Candlestick Chart for {self.selected_date}',
            xaxis_title='Time',
            yaxis_title='Price ($)',
            height=800,
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def create_trades_table(self):
        """Create an HTML table with trade information"""
        if not self.trades:
            return html.Div("No trades executed on this date")
        
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
                html.Td(f"{trade['entry_time'].strftime('%H:%M')} → {trade['exit_time'].strftime('%H:%M')}"),
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
        
        # Create the table
        table = html.Table(
            [header] + rows + [total_row],
            style={
                'width': '100%',
                'border-collapse': 'collapse',
                'margin-top': '20px',
                'margin-bottom': '20px'
            }
        )
        
        return table
    
    def create_summary_stats(self):
        """Create a summary statistics component"""
        if not self.trades:
            return html.Div("No trades executed on this date")
        
        total_profit = sum(trade['profit'] for trade in self.trades)
        total_profit_pct = sum(trade['profit_pct'] for trade in self.trades)
        avg_profit = total_profit / len(self.trades)
        avg_profit_pct = total_profit_pct / len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade['profit'] > 0)
        win_rate = (winning_trades / len(self.trades)) * 100
        
        return html.Div([
            html.H4(f"Trading Summary for {TICKER} on {self.selected_date}"),
            html.Ul([
                html.Li(f"Number of trades: {len(self.trades)}"),
                html.Li([
                    "Total profit: ",
                    html.Span(f"${total_profit:.2f} ({total_profit_pct:.2f}%)", 
                              style={'color': 'green' if total_profit > 0 else 'red'})
                ]),
                html.Li([
                    "Average profit per trade: ",
                    html.Span(f"${avg_profit:.2f} ({avg_profit_pct:.2f}%)",
                              style={'color': 'green' if avg_profit > 0 else 'red'})
                ]),
                html.Li(f"Win rate: {win_rate:.2f}% ({winning_trades}/{len(self.trades)})")
            ])
        ])
    
    def load_more_dates(self, days=None):
        """Load more historical dates"""
        if days is None:
            days = self.days * 2  # Double the number of days
        self.days = days
        self.available_dates = get_available_dates(days)
        return self.available_dates

# Create the Dash app
def create_dash_app():
    # Initialize the plotter
    plotter = DashPlotter()
    
    # Analyze data for the default date
    plotter.analyze_data()
    
    # Create the Dash app
    app = dash.Dash(__name__)
    
    # Define the app layout
    app.layout = html.Div([
        html.H1(f"{TICKER} Candlestick Analysis", style={'textAlign': 'center'}),
        
        # Date selection dropdown
        html.Div([
            html.Label("Select Date:"),
            dcc.Dropdown(
                id='date-dropdown',
                options=[{'label': date, 'value': date} for date in plotter.available_dates],
                value=plotter.selected_date,
                style={'width': '200px'}
            ),
            html.Button('Refresh Data', id='refresh-button', n_clicks=0, 
                       style={'marginLeft': '10px'}),
            html.Button('Load More Dates', id='more-dates-button', n_clicks=0,
                       style={'marginLeft': '10px'})
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),
        
        # Candlestick chart
        dcc.Graph(id='candlestick-chart', figure=plotter.create_figure()),
        
        # Trade summary and statistics
        html.Div([
            html.Div(id='trade-summary', children=plotter.create_summary_stats()),
            html.Div(id='trade-table', children=plotter.create_trades_table())
        ])
    ])
    
    # Define callback for date selection
    @app.callback(
        [Output('candlestick-chart', 'figure'),
         Output('trade-summary', 'children'),
         Output('trade-table', 'children')],
        [Input('date-dropdown', 'value'),
         Input('refresh-button', 'n_clicks')]
    )
    def update_chart(selected_date, n_clicks):
        # Analyze data for the selected date
        plotter.analyze_data(selected_date)
        
        # Create the figure and trade information
        fig = plotter.create_figure()
        summary = plotter.create_summary_stats()
        table = plotter.create_trades_table()
        
        return fig, summary, table
    
    # Define callback for loading more dates
    @app.callback(
        Output('date-dropdown', 'options'),
        [Input('more-dates-button', 'n_clicks')],
        [State('date-dropdown', 'value')]
    )
    def load_more_dates(n_clicks, current_value):
        if n_clicks > 0:
            # Load more dates
            plotter.load_more_dates()
            
            # Update dropdown options
            return [{'label': date, 'value': date} for date in plotter.available_dates]
        
        # Return current options if button not clicked
        return [{'label': date, 'value': date} for date in plotter.available_dates]
    
    return app

# Run the app
if __name__ == "__main__":
    app = create_dash_app()
    print(f"Starting Dash server for {TICKER} candlestick analysis...")
    print("Open your web browser and navigate to http://127.0.0.1:8050/")
    app.run_server(debug=True, use_reloader=False)
