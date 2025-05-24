import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(page_title="Stock Forecasting App", layout="wide")

# --- Logo ---
st.image("https://cdn-icons-png.flaticon.com/512/5974/5974636.png", width=60)
st.title("📈 Stock Market Forecasting Dashboard")

# --- Sidebar Inputs ---
st.sidebar.title("⚙️ Settings")
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
model_choice = st.sidebar.selectbox("Select Forecasting Model", ["ARIMA", "SARIMA", "Prophet", "LSTM", "LSTM Multivariate"])
stock_symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL")

# --- Theme Styling ---
if theme == "Dark":
    st.markdown("""
        <style>
            .stApp { background-color: #0e1117; color: white; }
            .css-1v0mbdj p, h1, h2, h3 { color: white; }
        </style>
    """, unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, start="2020-01-01", progress=False)
    df = df[["Close", "Volume"]].dropna()
    df.index.name = "Date"
    return df

df = load_data(stock_symbol)

# --- Forecast Models ---
def forecast_arima(df):
    model = ARIMA(df['Close'], order=(5, 1, 0))
    model_fit = model.fit()
    idx = pd.date_range(df.index[-1], periods=30, freq='B')
    return pd.Series(model_fit.forecast(steps=30), index=idx)

def forecast_sarima(df):
    model = SARIMAX(df['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    idx = pd.date_range(df.index[-1], periods=30, freq='B')
    return pd.Series(model_fit.forecast(steps=30), index=idx)

def forecast_prophet(df):
    data = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(data)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast.set_index('ds')['yhat'].tail(30)

def forecast_lstm(df):
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    input_seq = scaled[-60:].reshape(1, 60, 1)
    forecast = []
    for _ in range(30):
        pred = model.predict(input_seq)[0][0]
        forecast.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    idx = pd.date_range(df.index[-1], periods=30, freq='B')
    return pd.Series(forecast, index=idx)

def forecast_lstm_multivariate(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(60, len(scaled) - 30):
        X.append(scaled[i-60:i])
        y.append(scaled[i:i+30, 0])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(60, 2)),
        LSTM(64),
        Dense(30)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    last_seq = scaled[-60:].reshape(1, 60, 2)
    pred = model.predict(last_seq)[0]
    pred = scaler.inverse_transform(np.column_stack((pred, np.zeros(30))))[:, 0]
    idx = pd.date_range(df.index[-1], periods=30, freq='B')
    return pd.Series(pred, index=idx)

# --- Tabs Layout ---
tab1, tab2 = st.tabs(["📊 Data Overview", "🔮 Forecasting"])

with tab1:
    st.subheader(f"Historical Data for {stock_symbol}")
    st.line_chart(df["Close"])
    st.dataframe(df.tail(10), use_container_width=True)

with tab2:
    st.subheader(f"{model_choice} Forecast")

    if model_choice == "ARIMA":
        forecast = forecast_arima(df)
    elif model_choice == "SARIMA":
        forecast = forecast_sarima(df)
    elif model_choice == "Prophet":
        forecast = forecast_prophet(df)
    elif model_choice == "LSTM":
        forecast = forecast_lstm(df)
    else:
        forecast = forecast_lstm_multivariate(df)

    combined = pd.concat([df["Close"].iloc[-30:], forecast])
    max_point = combined.idxmax()
    min_point = combined.idxmin()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combined.index[:30], y=combined.values[:30], name="Last 30 Days"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast"))
    fig.add_trace(go.Scatter(x=[max_point], y=[combined.max()], name="Max", mode="markers+text",
                             marker=dict(color="green", size=10), text=["High"], textposition="top center"))
    fig.add_trace(go.Scatter(x=[min_point], y=[combined.min()], name="Min", mode="markers+text",
                             marker=dict(color="red", size=10), text=["Low"], textposition="bottom center"))
    fig.update_layout(title="📊 Price Forecast with Max/Min", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    st.metric("📈 Max Forecasted", f"${combined.max():.2f}")
    st.metric("📉 Min Forecasted", f"${combined.min():.2f}")
  with tab3:
    st.markdown("## ✅ Conclusion")
    st.write("""
This dashboard demonstrates how ARIMA, Prophet, and LSTM can be used to forecast stock prices from historical data.
Start with **Prophet** for balance. 
Use **LSTM** if data is large or nonlinear.  
Use this tool to quickly experiment with time series forecasting!

""")
    st.markdown("📘 **Developed by Mayakshi** · MSc.BDA Project · Powered by Streamlit")
