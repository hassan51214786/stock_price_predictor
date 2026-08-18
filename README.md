---

## 2. stock_price_predictor — README.md

```markdown
# 📈 Stock Price Predictor

A Streamlit app that pulls live stock data and predicts the next closing
price using a simple lagged linear regression model.

## How it works
- Fetches historical price data for any ticker via `yfinance`
- Builds lag features (previous 3 closing prices)
- Trains a Linear Regression model, evaluated with R² on unseen data
- Displays the closing price chart, last price, and predicted next price

## Try it
Enter any stock ticker (e.g. AAPL, TSLA, MSFT) to see recent price history
and a next-price prediction.

## Run locally
```bash
pip install streamlit yfinance pandas scikit-learn matplotlib
streamlit run stock_price_predictor.py