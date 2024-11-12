import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import pytz

# Function to load CSS
def load_css():
    css_file = "style.css"  # Assuming 'style.css' is in the same directory as the Streamlit script
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the CSS
load_css()

# App title and description
st.title("📈 Stock Price Predictor App")
st.markdown("""
This app uses a LSTM trained on NSE data to predict stock prices.
It also displays the stock's historical moving averages and technical indicators.
Popular Tickers: TATASTEEL.NS, ASIANPAINT.NS, TSLA, AAPL
""")

# User input for stock ticker
st.sidebar.header("Select Stock")
stock = st.sidebar.text_input("Enter a Stock Ticker", "^NSEI")

# timezone = pytz.utc
# Date settings
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Attempt to load stock data
try:
    data = yf.download(stock, start=start, end=end)
    # Remove the second row
    data = data.drop(data.index[1]).reset_index(drop=True)

    if data.empty:
        st.warning(f"No data found for the ticker symbol '{stock}'. Please try a different one.")
        st.stop()
    else:
        st.sidebar.success(f"Successfully loaded data for {stock}.")
        print(data.head())
except Exception as e:
    st.sidebar.error(f"Failed to download stock data: {e}")
    st.stop()

# Load pre-trained model
model_file = "model.keras"
if os.path.exists(model_file):
    try:
        model = load_model(model_file)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()
else:
    st.error("Model file not found. Please ensure 'model.keras' is in the same directory.")
    st.stop()


# Prediction using the entire data
x_test = pd.DataFrame(data['Close'])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test)

if len(scaled_data) >= 100:
    last_100_days = scaled_data[-100:].reshape(1, 100, 1)
    predicted_price = model.predict(last_100_days)
    predicted_price = scaler.inverse_transform(predicted_price)
    
    st.subheader("📊 Prediction for Tomorrow's Price/Points")
    st.write(f"**\*Rs. {predicted_price[0][0]:.2f}**")
else:
    st.warning("Not enough data points to perform prediction. Please select a longer date range or different stock ticker.")

# Stock data display
st.subheader("📅 Stock Data")
st.write(data.tail())

# Time period selection for graph
st.sidebar.header("Graph Time Period")
graph_period = st.sidebar.selectbox("Select Time Period for Graphs:", ["1 Year", "3 Years", "5 Years", "Max"])

# Settings for technical indicators
st.sidebar.header("Technical Indicator Settings")
ma_short = st.sidebar.number_input("Short-term MA Window:", min_value=1, max_value=200, value=10)
ma_long = st.sidebar.number_input("Long-term MA Window:", min_value=1, max_value=500, value=40)
rsi_window = st.sidebar.number_input("RSI Window:", min_value=1, max_value=100, value=20)
bb_window = st.sidebar.number_input("Bollinger Bands Window:", min_value=1, max_value=100, value=20)
bb_std_dev = st.sidebar.number_input("Bollinger Bands Std Dev:", min_value=1, max_value=3, value=2)
macd_short = st.sidebar.number_input("MACD Short Window:", min_value=1, max_value=300, value=50)
macd_long = st.sidebar.number_input("MACD Long Window:", min_value=1, max_value=300, value=100)
macd_signal = st.sidebar.number_input("MACD Signal Window:", min_value=1, max_value=50, value=9)
stochastic_rsi_window = st.sidebar.number_input("Stochastic RSI Window:", min_value=1, max_value=100, value=14)
k_period = st.sidebar.number_input("Stochastic RSI %K Period:", min_value=1, max_value=50, value=3)
d_period = st.sidebar.number_input("Stochastic RSI %D Period:", min_value=1, max_value=50, value=3)
williams_r_window = st.sidebar.number_input("Williams %R Window:", min_value=1, max_value=100, value=14)


# Filter data based on selected period (substracting no. of TRADING days)
if graph_period == "1 Year":
    data_filtered = data[-252:]
elif graph_period == "3 Years":
    data_filtered = data[-756:]
elif graph_period == "5 Years":
    data_filtered = data[-1260:]
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
data_filtered['BB_Upper'] = data_filtered['BB_Middle'] + (bb_std_dev * data_filtered['Close'].rolling(window=bb_window).std())
data_filtered['BB_Lower'] = data_filtered['BB_Middle'] - (bb_std_dev * data_filtered['Close'].rolling(window=bb_window).std())

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

# Additional Technical Indicators
def calculate_adx(data, window=14):
    plus_dm = data['High'].diff()
    minus_dm = -data['Low'].diff()
    tr = pd.concat([data['High'].diff(), data['Low'].diff(), data['Close'].diff().abs()], axis=1).max(axis=1)
    plus_di = 100 * (plus_dm.rolling(window=window).sum() / tr)
    minus_di = 100 * (minus_dm.rolling(window=window).sum() / tr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=window).mean()
    return adx

data_filtered['ADX'] = calculate_adx(data_filtered)

# Money Flow Index (MFI)
def calculate_mfi(data, window=14):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
    mfi = 100 * (positive_flow.rolling(window=window).sum() / negative_flow.rolling(window=window).sum())
    return mfi

data_filtered['MFI'] = calculate_mfi(data_filtered)

def calculate_stochastic_rsi(data, window=14, k_period=3, d_period=3):
    rsi = calculate_rsi(data, window)
    rsi_min = rsi.rolling(window=k_period).min()
    rsi_max = rsi.rolling(window=k_period).max()
    k = 100 * (rsi - rsi_min) / (rsi_max - rsi_min)
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_williams_r(data, window=14):
    high_high = data['High'].rolling(window=window).max()
    low_low = data['Low'].rolling(window=window).min()
    return -100 * (high_high - data['Close']) / (high_high - low_low)

data_filtered['%K'], data_filtered['%D'] = calculate_stochastic_rsi(data_filtered, stochastic_rsi_window, k_period, d_period)
data_filtered['Williams_R'] = calculate_williams_r(data_filtered, williams_r_window)

# Plotting and displaying stats below the graphs
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

# Create a new DataFrame row for the predicted price
predicted_date = data_filtered.index[-1] + pd.Timedelta(days=1)  # Add one day to the last date
predicted_data = pd.DataFrame({'Close': predicted_price[0][0]}, index=[predicted_date])

# Concatenate the predicted data with the original DataFrame
data_with_prediction = pd.concat([data_filtered, predicted_data])

# Plot the graph with the predicted price
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data_with_prediction['Close'][:-1], color='blue', label='Actual Price')  # Plot actual prices in blue
ax.plot(data_with_prediction['Close'][-1:], color='green', marker='o', markersize=10, label='Predicted Price')  # Plot predicted price in green

ax.set_xlabel('Date')
ax.set_ylabel('Price (Rs.)')
ax.set_title('Predicted Price vs. Actual Closing Price')
ax.legend()
st.pyplot(fig)
# Display Technical Indicators (without graphs)
st.header("📊 Key Technical Indicators")
st.write("Here are the latest values for key technical indicators:")
st.write(f"**ATR (Average True Range):** {data_filtered['ATR'].iloc[-1]:.2f}")
st.write(f"**OBV (On-Balance Volume):** {data_filtered['OBV'].iloc[-1]:.2f}")
st.write(f"**ADX (Average Directional Index):** {data_filtered['ADX'].iloc[-1]:.2f}")
st.write(f"**MFI (Money Flow Index):** {data_filtered['MFI'].iloc[-1]:.2f}")
st.write(f"**RSI (Relative Strength Index):** {data_filtered['RSI'].iloc[-1]:.2f}")


# Moving Averages Plot
plot_and_display_stats(
    "📊 Moving Average Crossover",
    "Price",
    {"Close Price": data_filtered['Close'], f"{ma_short}-day MA": data_filtered['MA_Short'], f"{ma_long}-day MA": data_filtered['MA_Long']},
    {"Short-term MA": data_filtered['MA_Short'].iloc[-1], "Long-term MA": data_filtered['MA_Long'].iloc[-1]}
)

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

# Stochastic RSI Plot
plot_and_display_stats(
    "Stochastic RSI",
    "RSI Value",
    {"%K": data_filtered['%K'], "%D": data_filtered['%D']},
    {"%K": data_filtered['%K'].iloc[-1], "%D": data_filtered['%D'].iloc[-1]}
)

# Williams %R Plot
plot_and_display_stats(
    "Williams %R",
    "Williams %R Value",
    {"Williams %R": data_filtered['Williams_R']},
    {"Williams %R": data_filtered['Williams_R'].iloc[-1]}
)





