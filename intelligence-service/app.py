import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Title
st.title("📦 Smart Inventory & Demand Forecasting for Retailers")

# Load and preprocess data
@st.cache_data
def load_data():
    sales = pd.read_csv('sales_train_validation.csv')
    calendar = pd.read_csv('calendar.csv')
    prices = pd.read_csv('sell_prices.csv')

    sales_long = sales.melt(id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                            var_name='d', value_name='sales')

    calendar['date'] = pd.to_datetime(calendar['date'])
    calendar = calendar[['d', 'wm_yr_wk', 'date']]
    df = sales_long.merge(calendar, on='d', how='left')
    df = df.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    df = df[['date', 'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'sales', 'sell_price']]
    return df

df = load_data()

# Sidebar
product_id = st.sidebar.selectbox("🔍 Select Product ID", df['id'].unique())
model_type = st.sidebar.radio("Select Model", ["Random Forest", "LSTM"])
current_inventory = st.sidebar.number_input("📦 Current Inventory", min_value=0, value=100)

# Filter product
product_data = df[df['id'] == product_id].sort_values('date')

if model_type == "Random Forest":
    product_data['lag_7'] = product_data['sales'].shift(7)
    product_data['lag_14'] = product_data['sales'].shift(14)
    product_data['lag_28'] = product_data['sales'].shift(28)
    product_data['rolling_mean_7'] = product_data['sales'].shift(1).rolling(window=7).mean()
    product_data['rolling_mean_14'] = product_data['sales'].shift(1).rolling(window=14).mean()
    product_data.fillna(0, inplace=True)

    features = ['lag_7', 'lag_14', 'rolling_mean_7', 'sell_price']
    X = product_data[features]
    y = product_data['sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    forecast = model.predict(X_test[-7:])
    forecasted_demand = np.sum(forecast)

    # Plot 1: Actual vs Predicted (Random Forest)
    st.subheader("📊 Random Forest - Actual vs Predicted")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(product_data['date'].iloc[-len(y_test):], y_test.values, label='Actual')
    ax1.plot(product_data['date'].iloc[-len(y_test):], y_pred, label='Predicted')
    ax1.set_title("Sales Forecast - Actual vs Predicted")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Units Sold")
    ax1.legend()
    ax1.grid()
    st.pyplot(fig1)

else:
    # LSTM Preprocessing
    df_lstm = product_data[['date', 'sales']].copy()
    df_lstm = df_lstm.sort_values('date')
    data = df_lstm['sales'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    SEQ_LEN = 30
    X, y = create_sequences(scaled_data, SEQ_LEN)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # LSTM Model
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(SEQ_LEN, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    y_pred = model.predict(X_test)
    last_seq = scaled_data[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
    future_predictions = []

    current_seq = last_seq
    for _ in range(7):
        pred = model.predict(current_seq)[0][0]
        future_predictions.append(pred)
        current_seq = np.append(current_seq[:, 1:, :], [[[pred]]], axis=1)

    forecasted_demand = np.sum(scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)))

    # Plot 2: LSTM - Actual vs Predicted
    st.subheader("📈 LSTM - Actual vs Predicted")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df_lstm['date'].values[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual')
    ax2.plot(df_lstm['date'].values[-len(y_test):], scaler.inverse_transform(y_pred), label='LSTM Predicted')
    ax2.set_title('LSTM Sales Forecast')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Sales')
    ax2.legend()
    ax2.grid()
    st.pyplot(fig2)

    # Plot 3: LSTM - Next 7-Day Forecast
    st.subheader("🔮 Next 7-Day Forecast (LSTM)")
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.plot(range(1, 8), scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)), marker='o')
    ax3.set_title("Next 7-Day Forecast (LSTM)")
    ax3.set_xlabel("Day")
    ax3.set_ylabel("Predicted Sales")
    ax3.grid()
    st.pyplot(fig3)

# Inventory check
st.markdown("### 🚨 Inventory Check")
if forecasted_demand > current_inventory:
    st.error(f"⚠️ Restock Alert! 7-day demand = {forecasted_demand:.0f}, Inventory = {current_inventory}")
else:
    st.success(f"✅ Inventory OK. 7-day demand = {forecasted_demand:.0f}, Inventory = {current_inventory}")

# Key metrics
st.markdown("### 📊 Summary")
st.write({
    '7-Day Forecasted Demand': round(forecasted_demand),
    'Current Inventory': current_inventory,
    'Gap': round(forecasted_demand - current_inventory)
})
