import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd

def get_spxl_intraday_data():
    """
    Retrieves 5-minute interval data for SPXL for the last 7 days
    """
    # Get today's date
    today = datetime.now().date()
    
    # Set start date to 7 days ago to ensure we have some data
    start_date = today - timedelta(days=7)
    end_date = today + timedelta(days=1)
    
    print(f"Fetching SPXL data from {start_date} to {end_date}")
    
    try:
        # Get the data with 5m intervals
        data = yf.download("SPXL", start=start_date, end=end_date, interval="5m", progress=False)
        
        print(f"Retrieved {len(data)} data points")
        print(f"Data columns: {data.columns.tolist()}")
        
        if not data.empty:
            print(f"Data range: {data.index.min()} to {data.index.max()}")
            print("\nSample data (first 5 rows):")
            print(data.head())
        else:
            print("WARNING: No data was retrieved!")
            
            # Try with a different ticker as a test
            print("\nTrying to fetch SPY data as a test...")
            spy_data = yf.download("SPY", start=start_date, end=end_date, interval="5m", progress=False)
            print(f"SPY data points: {len(spy_data)}")
            if not spy_data.empty:
                print("SPY data is available, so the issue might be specific to SPXL")
            
            # Try with a different interval
            print("\nTrying to fetch SPXL with daily interval...")
            daily_data = yf.download("SPXL", start=start_date, end=end_date, interval="1d", progress=False)
            print(f"SPXL daily data points: {len(daily_data)}")
            if not daily_data.empty:
                print("SPXL daily data is available, so the issue might be with intraday data")
                return daily_data  # Return daily data if intraday is not available
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    
    return data

def plot_candlestick(data):
    """
    Creates a candlestick chart using plotly
    """
    # Print column names to verify data structure
    print("\nData columns for plotting:", data.columns.tolist())
    
    # Check if we have the expected columns
    required_columns = ['Open', 'High', 'Low', 'Close']
    
    # Map column names if they're different (sometimes yfinance returns lowercase)
    column_map = {}
    for col in required_columns:
        if col in data.columns:
            column_map[col] = col
        elif col.lower() in data.columns:
            column_map[col] = col.lower()
    
    # Check if we have all required columns
    if len(column_map) < len(required_columns):
        print("WARNING: Missing some required columns for candlestick chart")
        print(f"Required: {required_columns}")
        print(f"Available: {data.columns.tolist()}")
        
        # Try to use the first 4 columns as OHLC if we can't find the expected columns
        if len(data.columns) >= 4:
            print("Using the first 4 columns as OHLC data")
            for i, col in enumerate(required_columns):
                column_map[col] = data.columns[i]
    
    print(f"Using column mapping: {column_map}")
    
    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data[column_map.get('Open', data.columns[0])],
        high=data[column_map.get('High', data.columns[1])],
        low=data[column_map.get('Low', data.columns[2])],
        close=data[column_map.get('Close', data.columns[3])],
        name='SPXL'
    )])
    
    # Update layout
    fig.update_layout(
        title='SPXL 5-Minute Candlestick Chart',
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    
    return fig

def main():
    # Get the data
    data = get_spxl_intraday_data()
    
    if data.empty:
        print("No data available. Market might be closed or data not yet available.")
        return
    
    # Print the full data to console for debugging
    print("\nFull data shape:", data.shape)
    print("\nData types:")
    print(data.dtypes)
    print("\nFirst few rows:")
    print(data.head())
    
    # Check if we have any NaN values
    print("\nNaN values in each column:")
    print(data.isna().sum())
    
    # Filter to just the most recent trading day with data
    if not data.empty:
        # Get the most recent date in the data
        latest_date = data.index.max().date()
        print(f"\nShowing data for {latest_date}")
        
        # Filter to just that date
        data = data[data.index.date == latest_date]
        print(f"Filtered to {len(data)} data points for {latest_date}")
        
        if len(data) == 0:
            print("WARNING: No data points for the latest date after filtering!")
            # Use all data if filtering resulted in empty dataset
            data = get_spxl_intraday_data()
            print(f"Using all available data instead: {len(data)} points")
    
    # Create the chart
    fig = plot_candlestick(data)
    
    # Show the chart
    fig.show()

if __name__ == "__main__":
    main()
