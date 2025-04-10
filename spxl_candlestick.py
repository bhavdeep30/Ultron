import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd

def get_spxl_intraday_data():
    """
    Retrieves 5-minute interval data for SPXL for the current day
    """
    # Get today's date
    today = datetime.now().date()
    
    # Set start date to today at market open (9:30 AM ET)
    # and end date to today at market close (4:00 PM ET)
    # Adding a day to end to ensure we get all of today's data
    start_date = today
    end_date = today + timedelta(days=1)
    
    # Get the data with 5m intervals
    data = yf.download("SPXL", start=start_date, end=end_date, interval="5m")
    
    return data

def plot_candlestick(data):
    """
    Creates a candlestick chart using plotly
    """
    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
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
        print("No data available for today. Market might be closed or data not yet available.")
        return
    
    # Create the chart
    fig = plot_candlestick(data)
    
    # Show the chart
    fig.show()

if __name__ == "__main__":
    main()
