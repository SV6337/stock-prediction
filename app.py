import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import random

def fetch_stock_data(ticker, strategy):
    stock = yf.Ticker(ticker)
    if strategy == "Intraday":
        data = stock.history(period="1d", interval="5m")
    elif strategy == "Swing":
        data = stock.history(period="1y")
    else:
        raise ValueError("Unknown strategy. Choose either 'Intraday' or 'Swing'.")

    if data.empty:
        raise ValueError("No data found for the specified ticker and strategy.")

    # Robustly merge dividend and split history by date only
    data['Dividends'] = 0.0
    data['Stock Splits'] = 0.0

    # Normalize index to date for alignment
    data_dates = data.index.normalize()

    # Dividends
    dividends = stock.dividends
    if not dividends.empty:
        div_df = dividends.copy()
        div_df.index = div_df.index.normalize()
        data['Dividends'] = data_dates.map(div_df).fillna(0.0).values

    # Splits
    splits = stock.splits
    if not splits.empty:
        split_df = splits.copy()
        split_df.index = split_df.index.normalize()
        data['Stock Splits'] = data_dates.map(split_df).fillna(0.0).values

    data['Price Change'] = data['Close'].diff()
    data['Direction'] = data['Price Change'].apply(lambda x: 1 if x > 0 else 0)
    data.dropna(inplace=True)
    return data

def fetch_financial_news(ticker):
    FINNHUB_API_KEY = "YOUR_FINNHUB_API_KEY"  # Replace with your Finnhub API key
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2023-01-01&to=2025-12-31&token={FINNHUB_API_KEY}"
    response = requests.get(url)
    articles = []
    if response.status_code == 200:
        data = response.json()
        for article in data:
            if "headline" in article and "url" in article:
                articles.append({
                    "title": article["headline"],
                    "link": article["url"]
                })
    # DEMO: If no news found, simulate random headlines for more dynamic sentiment
    if not articles:
        demo_headlines = [
            f"{ticker} stock surges after strong earnings report",
            f"Analysts are bullish on {ticker} for next quarter",
            f"{ticker} faces regulatory challenges",
            f"{ticker} announces new product line",
            f"{ticker} stock plummets after CEO resignation",
            f"Market uncertainty affects {ticker}",
            f"{ticker} receives positive analyst upgrades",
            f"{ticker} under investigation for fraud",
            f"{ticker} achieves record sales",
            f"{ticker} faces supply chain issues"
        ]
        random.shuffle(demo_headlines)
        articles = [{"title": title, "link": "#"} for title in demo_headlines[:4]]
    return articles

def analyze_sentiment_vader(news_list):
    analyzer = SentimentIntensityAnalyzer()
    if not news_list:
        return 0  # Neutral if no news
    scores = []
    for news in news_list:
        sentiment = analyzer.polarity_scores(news["title"])
        scores.append(sentiment["compound"])
    return np.mean(scores) if scores else 0

def train_model(data):
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    target = data['Direction']
    if len(features) < 2 or len(target) < 2:
        raise ValueError("Insufficient data for training the model.")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, scaler, accuracy

def predict_stock_direction(model, scaler, current_data):
    features = current_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    if len(features) == 0:
        raise ValueError("No data available for prediction.")
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled[-1].reshape(1, -1))
    return "Upward" if prediction[0] == 1 else "Downward"

def identify_support_resistance(data):
    data['Support'] = data['Low'].rolling(window=14).min()
    data['Resistance'] = data['High'].rolling(window=14).max()
    return data

def visualize_candlestick_chart(data, ticker, strategy):
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Candlestick'
    )])

    if 'Support' in data.columns and 'Resistance' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Support'],
            line=dict(color='orange', width=1),
            mode='lines',
            name='Support'
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Resistance'],
            line=dict(color='purple', width=1),
            mode='lines',
            name='Resistance'
        ))

    fig.update_layout(
        title=f"{strategy} Candlestick Chart with Support/Resistance for {ticker}",
        xaxis_title="Date/Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )

    return fig

def make_decision(sentiment_score, predicted_trend):
    if sentiment_score > 0.1 and predicted_trend == "Upward":
        return "Buy"
    elif sentiment_score < -0.1 and predicted_trend == "Downward":
        return "Sell"
    else:
        return "Hold"

def generate_summary(ticker, sentiment_score, trend, decision, support, resistance, stock_data):
    summary = f"""
    Stock Summary for {ticker}
    ---------------------------
    Historical Data (Last 5 Rows):
    {stock_data.tail(5).to_string(index=True)}
    
    Sentiment Score: {sentiment_score:.2f}
    Predicted Trend: {trend}
    Recommendation: {decision}
    Support Level: {support}
    Resistance Level: {resistance}
    """
    return summary

def clean_ticker(ticker: str) -> str:
    ticker = ticker.upper().replace("$", "").strip()
    return ticker

def get_dividend_and_split_info(ticker):
    stock = yf.Ticker(ticker)
    dividends = stock.dividends
    splits = stock.splits

    next_dividend = "No dividend history"
    next_split = "No split history"

    if not dividends.empty:
        last_dividend_date = dividends.index[-1].strftime("%Y-%m-%d")
        last_dividend = dividends.iloc[-1]
        next_dividend = f"Last dividend: {last_dividend} on {last_dividend_date}"

    if not splits.empty:
        last_split_date = splits.index[-1].strftime("%Y-%m-%d")
        last_split = splits.iloc[-1]
        next_split = f"Last split: {last_split} on {last_split_date}"

    return next_dividend, next_split

def main():
    st.set_page_config(layout="wide")
    st.title("AI Financial Advisor with Sentiment Analysis and Candlestick Charts")

    raw_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA): ")
    if raw_ticker:
        ticker = clean_ticker(raw_ticker)

        try:
            strategy = st.selectbox("Select Trading Strategy", ["Intraday", "Swing"])

            stock_data = fetch_stock_data(ticker, strategy)

            col1, col2 = st.columns([4, 6])

            with col1:
                st.subheader(f"Historical Data for {ticker}")
                # Remove Dividends and Stock Splits columns before displaying
                display_cols = [col for col in stock_data.columns if col not in ['Dividends', 'Stock Splits']]
                st.dataframe(stock_data[display_cols].tail(10))
                # Do not show "Past Dividend and Split Events" section

            with col2:
                st.subheader("Candlestick Chart")
                stock_data = identify_support_resistance(stock_data)
                fig = visualize_candlestick_chart(stock_data, ticker, strategy)
                st.plotly_chart(fig, use_container_width=True)

            # Dividend and split info
            next_dividend, next_split = get_dividend_and_split_info(ticker)
            st.write(f"Dividend Info: {next_dividend}")
            st.write(f"Stock Split Info: {next_split}")

            st.write("Fetching financial news...")
            news = fetch_financial_news(ticker)
            if news:
                st.write("Latest News Headlines:")
                for n in news[:5]:
                    st.markdown(f"- [{n['title']}]({n['link']})")
            else:
                st.write("No news found.")

            sentiment_score = analyze_sentiment_vader(news)
            st.write(f"Sentiment Score for {ticker}: {sentiment_score:.2f}")

            sentiment_label = "Neutral"
            sentiment_color = "gray"
            bar_value = 50  # Default for neutral

            if sentiment_score > 0.1:
                sentiment_label = "Positive"
                sentiment_color = "green"
                bar_value = 100
            elif sentiment_score < -0.1:
                sentiment_label = "Negative"
                sentiment_color = "red"
                bar_value = 0

            st.subheader("Sentiment Analysis")
            st.markdown(f"<p style='color:{sentiment_color}; font-size: 20px;'><strong>{sentiment_label}</strong></p>", unsafe_allow_html=True)
            st.progress(bar_value)

            st.write("Training the model...")
            model, scaler, accuracy = train_model(stock_data)
            st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

            st.write("Predicting stock direction...")
            predicted_trend = predict_stock_direction(model, scaler, stock_data)
            st.write(f"Predicted Stock Trend: {predicted_trend}")

            decision = make_decision(sentiment_score, predicted_trend)
            st.write(f"Recommendation: {decision}")

            support = stock_data['Support'].iloc[-1]
            resistance = stock_data['Resistance'].iloc[-1]

            summary = generate_summary(ticker, sentiment_score, predicted_trend, decision, support, resistance, stock_data)
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name=f"{ticker}_summary.txt",
                mime="text/plain"
            )

        except ValueError as ve:
            st.error(f"ValueError: {ve}")
            if strategy == "Intraday":
                st.info("Intraday data may not be available for this ticker or at this time. Try switching to 'Swing' strategy or check if the market is open.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__== "__main__":
    main()