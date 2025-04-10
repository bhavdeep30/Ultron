import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd

def get_spxl_data():
    """
    Retrieves data for SPXL for the last 30 days
    """
    # Get today's date
    today = datetime.now().date()
    
    # Set start date to 30 days ago to ensure we have data
    start_date = today - timedelta(days=30)
    end_date = today + timedelta(days=1)
    
    print(f"Fetching SPXL data from {start_date} to {end_date}")
    
    try:
        # Get daily data first (this should always work)
        data = yf.download("SPXL", start=start_date, end=end_date, interval="1d", progress=False)
        print(f"Retrieved {len(data)} daily data points")
        
        if not data.empty:
            print(f"Data range: {data.index.min()} to {data.index.max()}")
            print("\nSample data (first 5 rows):")
            print(data.head())
            return data
        else:
            # Try SPY as a fallback
            print("Trying SPY as a fallback...")
            spy_data = yf.download("SPY", start=start_date, end=end_date, interval="1d", progress=False)
            print(f"Retrieved {len(spy_data)} SPY data points")
            return spy_data
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        # Create some dummy data for testing
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dummy_data = pd.DataFrame({
            'Open': [100 + i for i in range(len(dates))],
            'High': [105 + i for i in range(len(dates))],
            'Low': [95 + i for i in range(len(dates))],
            'Close': [102 + i for i in range(len(dates))],
            'Volume': [1000000 for _ in range(len(dates))]
        }, index=dates)
        print("Created dummy data for testing")
        return dummy_data

def plot_stock_data(data):
    """
    Creates charts for the stock data
    """
    if 'Close' not in data.columns and 'close' in data.columns:
        data = data.rename(columns={'close': 'Close', 'open': 'Open', 
                                    'high': 'High', 'low': 'Low'})
    
    # Create a line chart of closing prices
    fig1 = px.line(data, x=data.index, y='Close', title='SPXL Closing Prices')
    fig1.update_layout(
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark'
    )
    
    # Create a candlestick chart if we have all required columns
    if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        fig2 = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='SPXL'
        )])
        
        fig2.update_layout(
            title='SPXL Candlestick Chart',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            xaxis_rangeslider_visible=False,
            template='plotly_dark'
        )
        
        return fig1, fig2
    
    return fig1, None

def main():
    # Get the data
    data = get_spxl_data()
    
    if data.empty:
        print("No data available.")
        return
    
    # Print data info
    print("\nData shape:", data.shape)
    print("Columns:", data.columns.tolist())
    
    # Create the charts
    fig1, fig2 = plot_stock_data(data)
    
    # Show the line chart (this should always work)
    fig1.show()
    
    # Show the candlestick chart if available
    if fig2 is not None:
        fig2.show()

if __name__ == "__main__":
    main()
