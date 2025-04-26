# Sentimental Analysis for Stock News

## Overview

This module analyzes the sentiment of news articles related to specific stock tickers using Yahoo Finance news data. The sentiment analysis is performed using a pre-trained FinBERT model from Hugging Face, which is fine-tuned for financial news. The sentiment is categorized as **positive**, **negative**, or **neutral** based on the content of the news article titles.

This tool can be used to help traders make data-driven decisions by evaluating the market sentiment surrounding stocks in real time.

## Features

- **Yahoo Finance News Scraping**: Automatically fetches the latest news articles for a given stock ticker.
- **Sentiment Analysis**: Uses the FinBERT model to classify the sentiment of news titles as **positive**, **negative**, or **neutral**.
- **Customizable Date Range**: Allows filtering of news articles based on specific date ranges.
- **Results Export**: Saves the sentiment analysis results to a CSV file for further analysis.

## Requirements

- Python 3.6+
- The following Python packages are required:
  - `yfinance` - to fetch stock news from Yahoo Finance.
  - `transformers` - for the FinBERT sentiment model.
  - `torch` - required by the Hugging Face model.
  - `pandas` - for data manipulation and CSV export.
