import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Stock AI Dashboard", layout="wide")

# -------------------------
# LOAD MODEL
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "linear_regression_model.pkl")
model = joblib.load(model_path)
st.write("✅ Model Loaded Successfully")
st.write("Model path:", model_path)

# -------------------------
# TITLE
# -------------------------
st.title("📈 Predict Future Stock Prices")

st.markdown("""
### AI-Powered Stock Market Analysis & Forecasting Platform

Analyze historical market trends, visualize technical indicators, and predict future stock prices using Machine Learning.
""")

# -------------------------
# STOCK SELECTOR
# -------------------------
st.sidebar.header("⚙ Settings")

stock = st.sidebar.selectbox(
    "Select Stock",
    (
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "META",
        "NVDA"
    )
)

period = st.sidebar.selectbox(
    "Time Period",
    (
        "6mo",
        "1y",
        "2y",
        "5y"
    )
)


# -------------------------
# LOAD DATA (IMPORTANT)
# -------------------------
with st.spinner("Loading stock data..."):
    df = yf.download(
        stock,
        period=period,
        auto_adjust=False,
        progress=False
    )
    st.write("Rows:", len(df))
st.write(df.tail())

# Fix MultiIndex columns from yfinance
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# -------------------------
# SAFETY CHECK
# -------------------------
if df.empty:
    st.error("No data found for selected stock")
    st.stop()

# -------------------------
# STOCK STATS
# -------------------------
st.subheader("📈 Stock Statistics")

col1, col2, col3 = st.columns(3)
st.write(df.head())
st.write(df.columns)
st.write(type(df["Close"]))
st.write(df["Close"].iloc[-1])
st.write(type(df["Close"].iloc[-1]))

current_price = float(df["Close"].iloc[-1])

col1.metric(
    "Current Price",
    f"${current_price:.2f}"
)
col2.metric("Highest Price", f"${df['High'].max():.2f}")
col3.metric("Lowest Price", f"${df['Low'].min():.2f}")

# -------------------------
# CANDLESTICK CHART
# -------------------------
st.subheader("🕯 Candlestick Chart")

fig = go.Figure(data=[
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )
])

fig.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# MOVING AVERAGE
# -------------------------
st.subheader("📊 Moving Average (20 Days)")

df["MA20"] = df["Close"].rolling(window=20).mean()
st.line_chart(df[["Close", "MA20"]])

# -------------------------
# PREDICTION SECTION
# -------------------------
st.subheader("🤖 AI Price Prediction")

st.write("Enter latest market values:")

open_price = st.number_input("Open Price")
high_price = st.number_input("High Price")
low_price = st.number_input("Low Price")
volume = st.number_input("Volume")

if st.button("Predict Closing Price"):

    input_data = np.array([[open_price, high_price, low_price, volume]])

    try:
        prediction = model.predict(input_data)
        st.success(f"📌 Predicted Closing Price: ${prediction[0]:.2f}")

    except Exception as e:
        st.error(e)