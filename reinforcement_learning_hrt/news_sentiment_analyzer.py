import yfinance as yf
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import json

def initialize_sentiment_model():
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def analyze_sentiment(text, tokenizer, model):
    if not text or str(text).lower() == 'nan':
        return "neutral"
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_idx = predictions.argmax().item()
    
    return ["neutral", "positive", "negative"][sentiment_idx]

def get_yahoo_news(ticker, max_news_items=50):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            print(f"No news found for {ticker}")
            return pd.DataFrame()
        
        # Process news items
        processed_news = []
        for item in news:
            try:
                # Extract relevant information from the nested structure
                title = item['content']['title'] if 'content' in item and 'title' in item['content'] else None
                publisher = item['content']['provider']['displayName'] if 'content' in item and 'provider' in item['content'] else None
                link = item['content']['canonicalUrl']['url'] if 'content' in item and 'canonicalUrl' in item['content'] else None
                pub_date = item['content']['pubDate'] if 'content' in item and 'pubDate' in item['content'] else None
                
                if title:  # Only include items with titles
                    processed_news.append({
                        'title': title,
                        'publisher': publisher,
                        'date': pd.to_datetime(pub_date) if pub_date else datetime.now(),
                        'link': link
                    })
            except Exception as e:
                print(f"Error processing news item: {e}")
                continue
        
        if not processed_news:
            print("No valid news items found after processing")
            return pd.DataFrame()
        
        return pd.DataFrame(processed_news).head(max_news_items)
    
    except Exception as e:
        print(f"Error fetching news: {e}")
        return pd.DataFrame()

def analyze_news_sentiment(ticker, start_date=None, end_date=None):
    tokenizer, model = initialize_sentiment_model()
    news_df = get_yahoo_news(ticker)
    
    if news_df.empty:
        return news_df
    
    # Filter by date if dates are provided
    if 'date' in news_df.columns and start_date and end_date:
        try:
            news_df = news_df[(news_df['date'] >= start_date) & 
                             (news_df['date'] <= end_date)]
        except Exception as e:
            print(f"Couldn't filter by date: {e}")
    
    if news_df.empty:
        print("No news after date filtering (if applied)")
        return news_df
    
    news_df['sentiment'] = news_df['title'].apply(
        lambda x: analyze_sentiment(x, tokenizer, model)
    )
    
    return news_df

if __name__ == "__main__":
    ticker = "AAPL"  # Try different tickers like "MSFT", "TSLA", "GOOG"
    
    # Try with and without date filters
    result_df = analyze_news_sentiment(
        ticker,
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )
    
    if not result_df.empty:
        print("\nNews Sentiment Analysis Results:")
        print(result_df[['date', 'title', 'sentiment', 'link']].to_string(index=False))
        
        csv_filename = f"{ticker}_news_sentiment.csv"
        result_df.to_csv(csv_filename, index=False)
        print(f"\nResults saved to {csv_filename}")
    else:
        print("No news available for analysis")
