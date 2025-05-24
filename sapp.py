import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime

st.set_page_config(page_title="Stock Forecasting Web App", layout="wide")
st.title("📈 Stock Forecasting Web App")

st.markdown("""
Welcome to the Stock Forecasting Web App! This app allows you to:
- Select a stock symbol (e.g., AAPL, TSLA)
- Choose from forecasting models: **ARIMA**, **SARIMA**, **Prophet**, or **LSTM**
- Generate and visualize a **30-day forecast**
""")

# Sidebar
symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
model_choice = st.sidebar.selectbox("Choose Forecasting Model", ["ARIMA", "SARIMA", "Prophet", "LSTM"])
n_days = st.sidebar.slider("Forecast Horizon (Days)", 10, 60, 30)

# Load data directly
data = yf.download(symbol, start="2020-01-01")
data.reset_index(inplace=True)
st.subheader(f"📊 Historical Stock Data: {symbol}")
st.write(data.tail())

# Forecast models
if model_choice == "ARIMA":
    st.header("ARIMA Forecast")
    data_arima = data.copy()
    data_arima.set_index("Date", inplace=True)
    model = ARIMA(data_arima["Close"], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=n_days)
    forecast_index = pd.date_range(data_arima.index[-1], periods=n_days + 1, freq='D')[1:]
    fig, ax = plt.subplots()
    ax.plot(data_arima["Close"], label="Historical")
    ax.plot(forecast_index, forecast, label="ARIMA Forecast", color='orange')
    ax.legend()
    st.pyplot(fig)
    st.success("🔍 **Conclusion (ARIMA):** Best for stable, linear trends. Simple and fast, but struggles with seasonality or volatility.")

elif model_choice == "SARIMA":
    st.header("SARIMA Forecast")
    data_sarima = data.copy()
    data_sarima.set_index("Date", inplace=True)
    model = SARIMAX(data_sarima["Close"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=n_days)
    forecast_index = pd.date_range(start=data_sarima.index[-1], periods=n_days+1, freq="D")[1:]
    fig, ax = plt.subplots()
    ax.plot(data_sarima["Close"], label="Historical")
    ax.plot(forecast_index, forecast, label="SARIMA Forecast", color='green')
    ax.legend()
    st.pyplot(fig)
    st.success("🔍 **Conclusion (SARIMA):** Ideal for seasonal patterns. More flexible than ARIMA, but requires fine-tuned parameters.")

elif model_choice == "Prophet":
    st.header("Prophet Forecast")
    prophet_df = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=n_days)
    forecast = model.predict(future)
    fig1 = model.plot(forecast)
    st.pyplot(fig1)
    st.success("🔍 **Conclusion (Prophet):** Great for capturing seasonality and trend changes. Easy to use, but may be less accurate on noisy financial data.")

elif model_choice == "LSTM":
    st.header("LSTM Forecast")
    data_lstm = data[["Date", "Close"]].copy()
    data_lstm.index = data_lstm["Date"]
    dataset = data_lstm["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    X_train, y_train = [], []
    for i in range(60, len(scaled_data) - n_days):
        X_train.append(scaled_data[i - 60:i])
        y_train.append(scaled_data[i:i + n_days, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_days))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    last_60 = scaled_data[-60:]
    input_seq = last_60.reshape(1, 60, 1)
    prediction = model.predict(input_seq)
    prediction = scaler.inverse_transform(prediction.reshape(-1, 1))

    forecast_dates = pd.date_range(data_lstm["Date"].iloc[-1], periods=n_days + 1)[1:]
    forecast_df = pd.DataFrame({"Date": forecast_dates, "LSTM_Predicted_Close": prediction.flatten()})

    fig, ax = plt.subplots()
    ax.plot(data_lstm["Date"], data_lstm["Close"], label="Historical")
    ax.plot(forecast_df["Date"], forecast_df["LSTM_Predicted_Close"], label="LSTM Forecast", color="red")
    ax.set_title(f"{symbol} - LSTM Forecast")
    ax.legend()
    st.pyplot(fig)
    st.success("🔍 **Conclusion (LSTM):** Best for volatile, non-linear patterns. Requires more data and compute, but delivers highly accurate forecasts.")