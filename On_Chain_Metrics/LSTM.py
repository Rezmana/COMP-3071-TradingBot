"""
onchain_train.py

This script fetches on-chain Bitcoin metrics from the CoinMetrics API,
performs feature engineering and preprocessing, creates time-windowed sequences,
builds and trains an LSTM model to predict the next day's log-transformed price,
and saves the trained model along with scalers and preprocessed data for later use.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from coinmetrics.api_client import CoinMetricsClient

# Set file paths and directories
output_dir_pkl = os.path.join(os.getcwd(), "data_pkl")
output_dir_onchain_output = os.path.join(os.getcwd(), "onchain_output")
os.makedirs(output_dir_pkl, exist_ok=True)

# Initialize CoinMetrics client
client = CoinMetricsClient()
print("CoinMetrics client initialized:", client)

# Fetch on-chain metrics for Bitcoin
metrics = client.get_asset_metrics(
    assets='btc',
    metrics=['PriceUSD','HashRate','TxTfrCnt','AdrActCnt','SplyCur'],
    start_time="2013-01-01",
    end_time="2023-01-02",
    frequency='1d'
)

# Convert the data to a DataFrame and save raw pickle file for record
metrics_df = pd.DataFrame(metrics)
raw_pickle_path = os.path.join(output_dir_pkl, "LSTM.pkl")
metrics_df.to_pickle(raw_pickle_path)
print("Raw on-chain data saved to:", raw_pickle_path)

# Copy DataFrame for preprocessing
df = metrics_df.copy()

# Convert time column to datetime and sort (assumes column named 'time')
df['time'] = pd.to_datetime(df['time'])
df.sort_values('time', inplace=True)

# Create target: next day's PriceUSD
df['PriceUSD_target'] = df['PriceUSD'].shift(-1)
df['PriceUSD_target'] = pd.to_numeric(df['PriceUSD_target'], errors='coerce')

# Create current and future price fields.
df['current_price'] = df['PriceUSD']
df['future_price'] = df['PriceUSD_target']

# Log-transform target to compress range
df['PriceUSD_log'] = np.log1p(df['PriceUSD_target'])

# Create technical indicators (moving averages, ratio, volatility)
df['MA_7'] = df['PriceUSD_target'].rolling(window=7).mean()
df['MA_30'] = df['PriceUSD_target'].rolling(window=30).mean()
df['MA_7_30_ratio'] = df['MA_7'] / df['MA_30']
df['Volatility_30'] = df['PriceUSD_target'].rolling(window=30).std()

# Drop NaN rows from shifting and rolling operations
df.dropna(inplace=True)

# Define features and target variables
features = ['HashRate', 'TxTfrCnt', 'AdrActCnt', 'SplyCur', 'PriceUSD_target', 'MA_7', 'MA_30', 'MA_7_30_ratio', 'Volatility_30']
X = df[features].values
y = df['PriceUSD_log'].values

# Scale features and target with MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Save the scalers for future use
import pickle
with open(os.path.join(output_dir_pkl, "scaler_X.pkl"), "wb") as f:
    pickle.dump(scaler_X, f)
with open(os.path.join(output_dir_pkl, "scaler_y.pkl"), "wb") as f:
    pickle.dump(scaler_y, f)
print("Scalers saved to", output_dir_pkl)

# Function to create time-windowed sequences
def create_sequences(X, y, time_steps=60):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

time_steps = 60
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# Split the sequences into train, validation, and test (70%-15%-15% time-series split)
X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# Build LSTM model
n_features = X_train.shape[2]
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Predict next-day price (log-transformed)

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
print("LSTM model built and compiled.")

# Set up early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# Save the trained model and training history
model_path = os.path.join(output_dir_onchain_output, "onchain_model_final.h5")
model.save(model_path)
print("Trained on-chain model saved to:", model_path)

with open(os.path.join(output_dir_pkl, "training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)
print("Training history saved.")

# Extract the columns needed for trading signal creation
price_data = df[['time', 'current_price', 'future_price']].copy()
# Save the integrated price data for later use (e.g., as a pickle file)
integrated_price_path = os.path.join(output_dir_pkl, 'integrated_price_data.pkl')
price_data.to_pickle(integrated_price_path)
print("Integrated price data saved to:", integrated_price_path)