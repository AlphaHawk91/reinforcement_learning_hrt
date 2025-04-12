import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler


def preprocess_data(config, file_path : str = "intraday_data.csv") :
    """ 
    
    """

    tickers = config["tickers"]

    df = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)

    # Flatten MultiIndex Columns: Convert ('Price', 'Ticker') → 'Price_Ticker'
    df.columns = [f"{col1}_{col2}" for col1, col2 in df.columns]

    # Convert Data Types: Ensure numeric columns are properly formatted
    df = df.apply(pd.to_numeric, errors='coerce')

    # Handle Missing Values (if any)
    df.ffill(inplace=True)  # Forward-fill missing values
    df.bfill(inplace=True)  # Backward-fill as a fallback

    # Ensure Index is a Datetime Index
    df.index.name = "Datetime"

    for ticker in tickers:
        # Calculate VWAP
        df[f'VWAP_{ticker}'] = (df[f'High_{ticker}'] + df[f'Low_{ticker}'] + df[f'Close_{ticker}']) / 3 * df[f'Volume_{ticker}']
        df[f'VWAP_{ticker}'] = df[f'VWAP_{ticker}'].cumsum() / df[f'Volume_{ticker}'].cumsum()

        # Compute daily forward return
        df[f'Forward_Return_{ticker}'] = df[f'Open_{ticker}'].pct_change(1).shift(-1)

        df.fillna({f'Forward_Return_{ticker}': 0 for ticker in tickers}, inplace=True)
        df.fillna({f'VWAP_{ticker}': 0 for ticker in tickers}, inplace=True)

    # Normalization using MinMaxScaler
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    # Save Cleaned Data for Analysis
    cleaned_file_path = "intraday_data_preprocessed.csv"
    df.to_csv(cleaned_file_path)

    print(f"""
    NaN Counts : {df.isna().sum().sum()}
    """)
    
    return cleaned_file_path