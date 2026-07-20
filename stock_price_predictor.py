import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Stock Predictor", page_icon="📈")
st.title("📈 Stock Price Predictor")
st.write("Predict future stock prices using Machine Learning")

ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA)", "AAPL")
data = yf.download(ticker, start="2020-01-01", end="2024-01-01")

assert data is not None
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

if data.empty:
    st.error("❌ Invalid ticker! Try something like AAPL, TSLA, MSFT")
else:
    st.subheader("📊 Recent Data")
    st.write(data.tail())

    st.subheader("📈 Closing Price Chart")
    fig, ax = plt.subplots()
    ax.plot(data['Close'], label="Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    df = data[['Close']].copy()
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df['Lag3'] = df['Close'].shift(3)
    df.dropna(inplace=True)

    X = df[['Lag1', 'Lag2', 'Lag3']]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    score = r2_score(y_test, test_pred)

    last_values = df[['Lag1', 'Lag2', 'Lag3']].iloc[-1].values.reshape(1, -1) # type: ignore
    prediction = model.predict(last_values) # type: ignore

    st.subheader("🤖 Prediction")
    st.write(f"**Model R² score (on unseen data): {score:.2f}**")

    col1, col2 = st.columns(2)
    with col1:
        last_price = float(data['Close'].iloc[-1])
        st.metric("📊 Last Price", f"${last_price:.2f}")
    with col2:
        pred_value = float(prediction[0])
        st.metric("🔮 Predicted Next Price", f"${pred_value:.2f}")

    st.info("⚠️ This is a simple ML model for learning purposes. Not for real trading decisions.")