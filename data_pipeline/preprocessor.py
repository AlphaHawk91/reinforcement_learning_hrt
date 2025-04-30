import pandas as pd
import numpy as np
# import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler


def preprocess_data(config, file_path : str = "intraday_data.csv", data_type : str = "training") -> str:
    """ 
    
    """

    tickers = config["tickers"]
    file_path = f"{data_type}_{file_path}"

    df = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)

    df.index.name = 'Date'
    df = df.reset_index()

    # Flattening MultiIndex Columns
    df.columns = [f"{feat}_{ticker}" if isinstance(feat, str) and feat != "Date" else feat for feat, ticker in df.columns]
    df_long = df.melt(id_vars='Date', var_name='Feature_Ticker', value_name='Value')
    df_long[['Feature', 'Ticker']] = df_long['Feature_Ticker'].str.rsplit('_', n=1, expand=True)
    df_long.drop(columns=['Feature_Ticker'], inplace=True)
    df_final = df_long.pivot(index=['Date', 'Ticker'], columns='Feature', values='Value').reset_index()
    df_final = df_final[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df_final.columns.name = None
    df_final.reset_index(drop=True, inplace=True)

    df = df_final.copy()

    # Handle Missing Values (if any)
    df.ffill(inplace=True)  # Forward-fill missing values
    df.bfill(inplace=True)  # Backward-fill as a fallback

    # Ensure Index is a Datetime Index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    df['VWAP'] = (df['High'] + df['Low'] + df['Close']) / 3

    df = df.sort_values(by=['Ticker', 'Date'])  # very important!
    df['Forward_Return'] = df.groupby('Ticker')['Close'].shift(-1) / df['Close'] - 1
    df['Forward_Return'] = df['Forward_Return'] * 100  # percent form

    df.fillna({'Forward_Return': 0}, inplace=True)
    df.fillna({'VWAP': 0}, inplace=True)

    # # Normalization using MinMaxScaler
    # scaler = MinMaxScaler()
    # column_range = df.columns[1:len(df.columns)-1]
    # df[column_range] = scaler.fit_transform(df[column_range])

    # Save Cleaned Data for Analysis
    cleaned_file_path = f"Processed_{file_path}"
    df.to_csv(cleaned_file_path)

    print(df)

    print(f"""
    NaN Counts : {df.isna().sum().sum()}
    """)
    
    # return cleaned_file_path