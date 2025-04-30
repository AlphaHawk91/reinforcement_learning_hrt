
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf


def download_data(config: dict, data_type: str) -> str:
    """
    Download historical stock data from Yahoo Finance in 8-day intervals.

    The data is saved to a CSV file named 'intraday_data.csv'.
    Returns the filename of the saved CSV.
    """
    tickers = config['tickers']
    interval = config['utility']['interval']

    
    start = datetime.strptime(config[data_type]['start'], "%Y-%m-%d")
    end = datetime.strptime(config[data_type]['end'], "%Y-%m-%d")

    # start = datetime.strptime(config['start'], "%Y-%m-%d")
    # end = datetime.strptime(config['end'], "%Y-%m-%d")

    # all_data = []

    data = yf.download(tickers, start=start, end=end, interval=interval)
    data.reset_index(inplace=True)

    data.to_csv(f'{data_type}_intraday_data.csv', index=False)

        # # Generate 8-day intervals
        # current_start = start
        # while current_start < end:
        #     current_end = min(current_start + timedelta(days=7), end)
        #     print(f"Downloading data from {current_start.date()} to {current_end.date()}...")

        #     # Download data for the current interval
        #     data = yf.download(tickers, start=current_start,
        #                     end=current_end, interval=interval)
        #     data.reset_index(inplace=True)
        #     all_data.append(data)

        #     # Move to the next interval
        #     current_start = current_end

        # # Concatenate all data into a single DataFrame
        # if all_data:
        #     final_data = pd.concat(all_data, ignore_index=True)
        # else:
        #     final_data = pd.DataFrame()

        # final_data.to_csv('intraday_data.csv', index=False)

        # return 'intraday_data.csv'
