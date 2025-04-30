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
    
    # print(df.columns)

    df.index.name = 'Date'
    df = df.reset_index()

    # Step 2: Flatten MultiIndex columns to normal columns
    df.columns = [f"{feat}_{ticker}" if isinstance(feat, str) and feat != "Date" else feat for feat, ticker in df.columns]

    # Step 3: Reshape to long format
    df_long = df.melt(id_vars='Date', var_name='Feature_Ticker', value_name='Value')

    # Step 4: Split the combined column into two: Feature and Ticker
    df_long[['Feature', 'Ticker']] = df_long['Feature_Ticker'].str.rsplit('_', n=1, expand=True)
    df_long.drop(columns=['Feature_Ticker'], inplace=True)

    # Step 5: Pivot so each row is (Date, Ticker) and each feature is a column
    df_final = df_long.pivot(index=['Date', 'Ticker'], columns='Feature', values='Value').reset_index()

    # Step 6: Optional: reorder columns
    df_final = df_final[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df_final.columns.name = None
    # df_final.drop(columns=['Feature'], inplace=True)

    df_final.reset_index(drop=True, inplace=True)
    # df.index.name = 'Date'
    # df_final.set_index('Date', inplace=True)
    # df_final.index = pd.to_datetime(df_final.index)

    # print(df_final.columns)
    # print(df_final.head())


    file_path = f"Processed_{file_path}"


    df_final.to_csv(file_path, index=False)
    print(f"Data preprocessed and saved to {file_path}.")

    return file_path



    # # Flatten MultiIndex Columns: Convert ('Price', 'Ticker') → 'Price_Ticker'
    # df.columns = [f"{col1}_{col2}" for col1, col2 in df.columns]

    # # Convert Data Types: Ensure numeric columns are properly formatted
    # df = df.apply(pd.to_numeric, errors='coerce')

    # # Handle Missing Values (if any)
    # df.ffill(inplace=True)  # Forward-fill missing values
    # df.bfill(inplace=True)  # Backward-fill as a fallback

    # # Ensure Index is a Datetime Index
    # df.index.name = "Datetime"

    # for ticker in tickers:
    #     # Calculate VWAP
    #     df[f'VWAP_{ticker}'] = (df[f'High_{ticker}'] + df[f'Low_{ticker}'] + df[f'Close_{ticker}']) / 3 * df[f'Volume_{ticker}']
    #     df[f'VWAP_{ticker}'] = df[f'VWAP_{ticker}'].cumsum() / df[f'Volume_{ticker}'].cumsum()

    #     # Compute daily forward return
    #     df[f'Forward_Return_{ticker}'] = df[f'Open_{ticker}'].pct_change(1).shift(-1)

    #     df.fillna({f'Forward_Return_{ticker}': 0 for ticker in tickers}, inplace=True)
    #     df.fillna({f'VWAP_{ticker}': 0 for ticker in tickers}, inplace=True)

    # # Normalization using MinMaxScaler
    # scaler = MinMaxScaler()
    # df[df.columns] = scaler.fit_transform(df[df.columns])

    # # Save Cleaned Data for Analysis
    # cleaned_file_path = "intraday_data_preprocessed.csv"
    # df.to_csv(cleaned_file_path)

    # print(f"""
    # NaN Counts : {df.isna().sum().sum()}
    # """)
    
    # return cleaned_file_path