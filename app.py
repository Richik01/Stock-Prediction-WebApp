import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# App title and description
st.title("📈 Stock Price Predictor App")
st.markdown("""
This app uses a pre-trained model to predict stock prices.
It also displays the stock's historical moving averages and technical indicators.
""")

# User input for stock ticker
st.sidebar.header("Stock Selection")
stock = st.sidebar.text_input("Enter the Stock Ticker (e.g., AAPL, MSFT)", "^NSEI")

# Date settings
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Attempt to load stock data
try:
    data = yf.download(stock, start=start, end=end)
    if data.empty:
        st.warning(f"No data found for the ticker symbol '{stock}'. Please try a different one.")
        st.stop()
    else:
        st.sidebar.success(f"Successfully loaded data for {stock}.")
except Exception as e:
    st.sidebar.error(f"Failed to download stock data: {e}")
    st.stop()

# Load pre-trained model
try:
    model = load_model("model.keras")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Prediction using the entire data
x_test = pd.DataFrame(data['Close'])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test)

if len(scaled_data) >= 100:
    last_100_days = scaled_data[-100:].reshape(1, 100, 1)
    predicted_price = model.predict(last_100_days)
    predicted_price = scaler.inverse_transform(predicted_price)
    
    st.subheader("📊 Prediction for Tomorrow's Price")
    st.write(f"**${predicted_price[0][0]:.2f}**")
else:
    st.warning("Not enough data points to perform prediction. Please select a longer date range or different stock ticker.")

# Stock data display
st.subheader("📅 Stock Data")
st.write(data.tail())

# Settings for technical indicators
st.sidebar.header("Technical Indicator Settings")
ma_short = st.sidebar.number_input("Short-term MA Window:", min_value=1, max_value=200, value=50)
ma_long = st.sidebar.number_input("Long-term MA Window:", min_value=1, max_value=500, value=200)
rsi_window = st.sidebar.number_input("RSI Window:", min_value=1, max_value=100, value=14)
bb_window = st.sidebar.number_input("Bollinger Bands Window:", min_value=1, max_value=100, value=20)
bb_std_dev = st.sidebar.number_input("Bollinger Bands Std Dev:", min_value=1, max_value=3, value=2)
macd_short = st.sidebar.number_input("MACD Short Window:", min_value=1, max_value=50, value=12)
macd_long = st.sidebar.number_input("MACD Long Window:", min_value=1, max_value=50, value=26)
macd_signal = st.sidebar.number_input("MACD Signal Window:", min_value=1, max_value=50, value=9)

# Time period selection for graph
st.sidebar.header("Graph Time Period")
graph_period = st.sidebar.selectbox("Select Time Period for Graphs:", ["1 Year", "3 Years", "5 Years", "Max"])

# Filter data based on selected period
if graph_period == "1 Year":
    data_filtered = data[-252:]  # Approx. last 252 trading days
elif graph_period == "3 Years":
    data_filtered = data[-756:]  # Approx. last 756 trading days
elif graph_period == "5 Years":
    data_filtered = data[-1260:]  # Approx. last 1260 trading days
else:
    data_filtered = data  # Use entire data

# Calculate Moving Averages
data_filtered['MA_Short'] = data_filtered['Close'].rolling(window=ma_short).mean()
data_filtered['MA_Long'] = data_filtered['Close'].rolling(window=ma_long).mean()

# Relative Strength Index (RSI)
def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

data_filtered['RSI'] = calculate_rsi(data_filtered, rsi_window)

# Bollinger Bands
data_filtered['BB_Middle'] = data_filtered['Close'].rolling(window=bb_window).mean()
data_filtered['BB_Upper'] = data_filtered['BB_Middle'] + bb_std_dev * data_filtered['Close'].rolling(window=bb_window).std()
data_filtered['BB_Lower'] = data_filtered['BB_Middle'] - bb_std_dev * data_filtered['Close'].rolling(window=bb_window).std()

# MACD
data_filtered['EMA_Short'] = data_filtered['Close'].ewm(span=macd_short, adjust=False).mean()
data_filtered['EMA_Long'] = data_filtered['Close'].ewm(span=macd_long, adjust=False).mean()
data_filtered['MACD'] = data_filtered['EMA_Short'] - data_filtered['EMA_Long']
data_filtered['Signal_Line'] = data_filtered['MACD'].ewm(span=macd_signal, adjust=False).mean()

# Average True Range (ATR)
def calculate_atr(data, window=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()

data_filtered['ATR'] = calculate_atr(data_filtered)

# On-Balance Volume (OBV)
data_filtered['OBV'] = (np.sign(data_filtered['Close'].diff()) * data_filtered['Volume']).cumsum()

# Plotting with statistics below each graph
def plot_and_display_stats(title, y_label, data, stats_dict):
    st.subheader(title)
    fig, ax = plt.subplots(figsize=(15, 6))
    for label, series in data.items():
        ax.plot(series, label=label)
    ax.set_ylabel(y_label)
    ax.legend()
    st.pyplot(fig)
    for stat, value in stats_dict.items():
        st.write(f"**{stat}:** {value:.2f}")

# Moving Averages Plot
plot_and_display_stats(
    "📊 Moving Averages",
    "Price",
    {"Close Price": data_filtered['Close'], f"{ma_short}-day MA": data_filtered['MA_Short'], f"{ma_long}-day MA": data_filtered['MA_Long']},
    {"Short-term MA": data_filtered['MA_Short'].iloc[-1], "Long-term MA": data_filtered['MA_Long'].iloc[-1]}
)

# RSI Plot
plot_and_display_stats(
    "📈 Relative Strength Index (RSI)",
    "RSI Value",
    {"RSI": data_filtered['RSI']},
    {"RSI Value": data_filtered['RSI'].iloc[-1]}
)

# Bollinger Bands Plot
plot_and_display_stats(
    "📉 Bollinger Bands",
    "Price",
    {"Close Price": data_filtered['Close'], "Upper Band": data_filtered['BB_Upper'], "Middle Band": data_filtered['BB_Middle'], "Lower Band": data_filtered['BB_Lower']},
    {"Upper Band": data_filtered['BB_Upper'].iloc[-1], "Middle Band": data_filtered['BB_Middle'].iloc[-1], "Lower Band": data_filtered['BB_Lower'].iloc[-1]}
)

# MACD Plot
plot_and_display_stats(
    "📉 MACD with Signal Line",
    "MACD Value",
    {"MACD": data_filtered['MACD'], "Signal Line": data_filtered['Signal_Line']},
    {"MACD": data_filtered['MACD'].iloc[-1], "Signal Line": data_filtered['Signal_Line'].iloc[-1]}
)

# ATR Plot
plot_and_display_stats(
    "📈 Average True Range (ATR)",
    "ATR Value",
    {"ATR": data_filtered['ATR']},
    {"ATR Value": data_filtered['ATR'].iloc[-1]}
)

# OBV Plot
plot_and_display_stats(
    "📈 On-Balance Volume (OBV)",
    "OBV Value",
    {"OBV": data_filtered['OBV']},
    {"OBV": data_filtered['OBV'].iloc[-1]}
)

# Preparing and plotting original vs predicted prices
scaled_data = scaler.fit_transform(pd.DataFrame(data['Close']))
x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)
predictions = model.predict(x_data)
inv_predictions = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Plot original vs predicted values
plotting_data = pd.DataFrame({
    "Original": inv_y_test.reshape(-1),
    "Predicted": inv_predictions.reshape(-1)
}, index=data.index[100:])

st.subheader("📉 Original vs Predicted Close Prices")
fig_pred, ax_pred = plt.subplots(figsize=(15, 6))
ax_pred.plot(data['Close'], label="Actual Data", color="blue", linewidth=1.5)
ax_pred.plot(plotting_data['Original'], label="Original Test Data", color="green", linestyle="--")
ax_pred.plot(plotting_data['Predicted'], label="Predicted Data", color="red", linestyle="--")
ax_pred.legend()
st.pyplot(fig_pred)
